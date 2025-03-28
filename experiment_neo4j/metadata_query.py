import json
import argparse
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate

# --- CONFIG ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "harsh@123"

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the LLM
OLLAMA_MODEL = "granite3.2:latest"
llm = Ollama(model=OLLAMA_MODEL)

# Schema extraction prompt
schema_extraction_prompt = PromptTemplate.from_template("""
You are an AI assistant with expertise in database metadata. You help generate schema information based on natural language queries from users.

## Capabilities  
- Interpret user queries about data
- Generate JSON schema information
- Identify relationships between tables and columns

## Context
The user wants to extract data for the following query:
"{query}"

## Available Tables and Columns
The following tables and columns are relevant to the query:
{table_info}

## Task
1. Identify which tables and columns are needed to answer the query
2. Create a JSON schema that would be suitable to extract the necessary data
3. Format the schema to include table names, column names, and any relationships

Return only the JSON schema without explanation or introduction. The schema should have the structure:
{{
  "tables": [
    {{
      "name": "table_name",
      "schema": "schema_name",
      "columns": ["column1", "column2", ...],
      "description": "description of the table"
    }},
    ...
  ],
  "relationships": [
    {{
      "source": "schema1.table1.column1",
      "target": "schema2.table2.column2",
      "description": "description of relationship"
    }},
    ...
  ]
}}
""")

def get_relevant_tables_and_columns(query, top_n=3):
    """
    Find tables and columns relevant to the user query using semantic search
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    query_embedding = embedding_model.encode(query)
    
    with driver.session() as session:
        # Find relevant tables
        tables_result = session.run("""
            MATCH (t:DataObject)
            WHERE t.name IS NOT NULL AND t.schema IS NOT NULL
            RETURN t.name AS name, t.schema AS schema, t.sourceSystem AS source_system, 
                   id(t) AS id, t.description AS description
        """)
        
        tables = []
        for record in tables_result:
            tables.append({
                "name": record["name"],
                "schema": record["schema"], 
                "source_system": record["source_system"],
                "id": record["id"],
                "description": record["description"] if record["description"] else "",
                "relevance": 0.0  # Will be updated with embedding similarity
            })
        
        # Calculate relevance for tables
        for table in tables:
            table_text = f"{table['schema']}.{table['name']} {table['description']}"
            table_embedding = embedding_model.encode(table_text)
            similarity = np.dot(query_embedding, table_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(table_embedding))
            table["relevance"] = float(similarity)
        
        # Sort tables by relevance and take top n
        tables = sorted(tables, key=lambda x: x["relevance"], reverse=True)[:top_n]
        
        # Get columns for the top tables
        all_columns = []
        for table in tables:
            columns_result = session.run("""
                MATCH (t:DataObject)-[:HAS_COLUMN]->(c:Column)
                WHERE id(t) = $table_id
                OPTIONAL MATCH (c)-[:HAS_CONCEPT]->(concept:Concept)
                RETURN c.name AS name, c.dataType AS data_type, c.description AS description,
                       concept.name AS concept
            """, table_id=table["id"])
            
            columns = []
            for col_record in columns_result:
                columns.append({
                    "name": col_record["name"],
                    "data_type": col_record["data_type"],
                    "description": col_record["description"] if col_record["description"] else "",
                    "concept": col_record["concept"] if col_record["concept"] else "",
                    "relevance": 0.0  # Will be updated with embedding similarity
                })
            
            # Calculate relevance for columns
            for column in columns:
                column_text = f"{column['name']} {column['description']} {column['concept']}"
                column_embedding = embedding_model.encode(column_text)
                similarity = np.dot(query_embedding, column_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(column_embedding))
                column["relevance"] = float(similarity)
            
            # Sort columns by relevance
            columns = sorted(columns, key=lambda x: x["relevance"], reverse=True)
            table["columns"] = columns
            
            # Add to all columns list for relationship finding
            all_columns.extend([{**col, "table": table["name"], "schema": table["schema"]} for col in columns])
        
        # Find relationships between the relevant tables
        relationships = []
        if len(tables) > 1:
            for i in range(len(tables)):
                for j in range(i+1, len(tables)):
                    rel_result = session.run("""
                        MATCH (t1:DataObject)-[:HAS_COLUMN]->(c1:Column)-[r:POSSIBLE_FOREIGN_KEY]->(c2:Column)<-[:HAS_COLUMN]-(t2:DataObject)
                        WHERE id(t1) = $table1_id AND id(t2) = $table2_id
                        RETURN c1.name AS source_col, c2.name AS target_col
                    """, table1_id=tables[i]["id"], table2_id=tables[j]["id"])
                    
                    for rel in rel_result:
                        relationships.append({
                            "source": f"{tables[i]['schema']}.{tables[i]['name']}.{rel['source_col']}",
                            "target": f"{tables[j]['schema']}.{tables[j]['name']}.{rel['target_col']}",
                            "description": f"Foreign key relationship from {tables[i]['name']}.{rel['source_col']} to {tables[j]['name']}.{rel['target_col']}"
                        })
    
    driver.close()
    return tables, relationships

def generate_schema_for_query(query):
    """
    Generate a schema based on the user's natural language query
    """
    # Get relevant tables and columns
    tables, relationships = get_relevant_tables_and_columns(query)
    
    # Format table info for the prompt
    table_info = ""
    for table in tables:
        table_info += f"Table: {table['schema']}.{table['name']}\n"
        table_info += f"Description: {table['description']}\n"
        table_info += "Columns:\n"
        
        for col in table['columns'][:10]:  # Top 10 most relevant columns
            table_info += f"  - {col['name']} ({col['data_type']}): {col['description']}\n"
        
        table_info += "\n"
    
    # Format relationship info
    if relationships:
        table_info += "Relationships:\n"
        for rel in relationships:
            table_info += f"  - {rel['source']} → {rel['target']}\n"
    
    # Generate schema using LLM
    prompt = schema_extraction_prompt.format(query=query, table_info=table_info)
    schema_json = llm.invoke(prompt)
    
    # Parse and format the JSON
    try:
        schema = json.loads(schema_json)
        return schema
    except json.JSONDecodeError:
        # If the LLM response is not valid JSON, try to extract just the JSON part
        try:
            # Find JSON-like content between curly braces
            start = schema_json.find('{')
            end = schema_json.rfind('}') + 1
            if start >= 0 and end > start:
                extracted_json = schema_json[start:end]
                schema = json.loads(extracted_json)
                return schema
        except:
            pass
        
        # If still failing, return a basic structure with the raw tables and relationships
        return {
            "tables": [
                {
                    "name": table["name"],
                    "schema": table["schema"],
                    "columns": [col["name"] for col in table["columns"][:5]],
                    "description": table["description"]
                }
                for table in tables
            ],
            "relationships": relationships,
            "error": "Could not generate valid JSON schema from LLM response"
        }

def main():
    # Hard-coded test queries
    test_queries = [
        "Show me the OEE trend for Line 3 over the past quarter",
        # "Which production line has the highest throughput for Consumer Goods products?"
    ]
    
    for query in test_queries:
        print(f"Generating schema for query: '{query}'")
        schema = generate_schema_for_query(query)
        
        # Pretty print the schema
        schema_json = json.dumps(schema, indent=2)
        print("\nGenerated Schema:")
        print(schema_json)
        
        # Save to file
        output_file = f"{query.replace(' ', '_')[:20].lower()}.json"
        with open(output_file, 'w') as f:
            f.write(schema_json)
        print(f"\nSchema saved to {output_file}")

if __name__ == "__main__":
    main()
