import psycopg2
from neo4j import GraphDatabase
import ssl
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "harsh@123"

PG_CONN = {
    "host": "insights-db.postgres.database.azure.com",
    "dbname": "turinton_backend_stage",
    "port": "5432",
    "user": "turintonadmin",
    "password": "Passw0rd123!", 
    "sslmode": "require"
}

OLLAMA_MODEL = "granite3.2:latest"

llm = Ollama(model=OLLAMA_MODEL)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

prompt = PromptTemplate.from_template("""
You are an AI assistant with knowledge of ontologies. You help map database columns to manufacturing-specific business concepts.

## Capabilities  
- Interpret column metadata and sample data  
- Map columns to standard manufacturing and supply chain ontologies  
- Recognize business concepts in operational and production systems  

## Response Guidelines  
- Use only the provided metadata and sample data  
- Return only the ontology name (e.g., Equipment, WorkOrder, Product, Batch, Shift, Inventory, Region, etc.)  
- Do not include any explanation or extra text  
- Return a single ontology term or "None" if no match  

## Key Priorities  
- Prioritize manufacturing and supply chain concepts  
- Use sample data and descriptions for disambiguation  
- Be accurate and deterministic  

## Response Flow  
1. Analyze column metadata and context  
2. Identify the closest manufacturing ontology concept  
3. Return the ontology name or "None"  

---

**Task**:  
Given the following metadata from a manufacturing database:

- Column name: `{col_name}`  
- Data type: `{dtype}`  
- Table: `{table}`  
- Schema: `{schema}`  
- Description: `{description}`  
- Sample data: `{sample_data}`  

Return the most appropriate **manufacturing ontology concept** this column represents.  
Respond with the ontology name only.
""")

hierarchy_prompt = PromptTemplate.from_template("""
You are an AI assistant with knowledge of ontologies that helps organize business concepts into hierarchies in the manufacturing and supply chain domain.

## Capabilities  
- Identify and classify business concepts  
- Map concepts to standard manufacturing and supply chain ontologies  
- Return parent-child relationships for business terms  

## Response Guidelines  
- Use ontology data only  
- Be concise and precise  
- Do not provide suggestions or explanations  
- Respond with only the parent concept or "None"  
- Do not answer if the concept is outside domain scope  

## Key Priorities  
- Maintain accuracy in concept hierarchy  
- Prefer specific domain ontologies over generic ones  
- Flag ambiguous or unmatched concepts with "None"  

## Response Flow  
1. Analyze concept  
2. Match against standard ontologies  
3. Return most specific parent concept or "None"  

---

**Task**:  
Given the concept: `{concept}`  
Return its **most specific parent concept** from a standard **Manufacturing or Supply Chain ontology**, or `"None"` if not found.  
Respond with the parent concept only.
""")

foreign_key_prompt = PromptTemplate.from_template("""
You are an AI assistant with expertise in databases. You help identify likely foreign key relationships between tables in structured data systems.

## Capabilities  
- Analyze table schemas and column names  
- Infer likely foreign key relationships  
- Suggest join paths between tables  

## Response Guidelines  
- Use only the provided schema and column data  
- Be concise and deterministic  
- Do not make assumptions beyond the data  
- Format result as: `<schema1>.<table1>.<column1> -> <schema2>.<table2>.<column2>`  
- If no relationship is found, return `"None"`  
- No suggestions or explanations  

## Key Priorities  
- Prioritize exact or semantically similar column name matches  
- Favor primary key–foreign key patterns  
- Avoid false positives if match is uncertain  

## Response Flow  
1. Analyze table and column names  
2. Identify potential foreign key relationships  
3. Return most likely column pair or `"None"`  

---

**Task**:  
Given the following table schemas, determine if a foreign key relationship likely exists.  

Table 1: `{table1_name}` with columns: `{table1_columns}`  
Table 2: `{table2_name}` with columns: `{table2_columns}`  

Return the relationship as:  
`<schema1>.<table1>.<column1> -> <schema2>.<table2>.<column2>`  
or `"None"` if no likely relationship is found.
""")

def load_metadata_into_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()
    print("\U0001F9F9 Cleared existing Neo4j data.")

    conn = psycopg2.connect(**PG_CONN)
    cursor = conn.cursor()

    cursor.execute("SELECT table_name, database_name, db_type, datasource_id, id, enriched_metadata, sample_data FROM public.metadata where datasource_id='d8314b21-6456-478d-bc81-fcbd898eec1a'")
    rows = cursor.fetchall()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for row in rows:
            table_name, schema_name, source_system, datasource_id, table_id, column_metadata_json, sample_data = row

            try:
                session.write_transaction(create_dataobject, table_name, schema_name, source_system, datasource_id, table_id)

                db_meta = column_metadata_json.get("database_metadata", {})

                for schema_name, tables in db_meta.items():
                    for table_name, views in tables.items():
                        for view in views:
                            column_metadata = view.get("column_metadata", [])
                            for col in column_metadata:
                                col_name = col.get("Field")
                                data_type = col.get("Type")
                                is_nullable = col.get("is_nullable")
                                comment = col.get("Comments")
                                description = col.get("description")
                                sample_data = col.get("sample_values")

                                concept = classify_column_with_llm(col_name, data_type, table_name, schema_name, description, sample_data)
                                embedding = embedding_model.encode(col_name).tolist()

                                session.write_transaction(
                                    create_column_and_relationship,
                                    table_name,
                                    schema_name,
                                    source_system,
                                    col_name,
                                    data_type,
                                    is_nullable,
                                    comment,
                                    table_id,
                                    concept,
                                    embedding,
                                    embedding_model.encode(concept).tolist()
                                )

                                parent_concept = get_parent_concept_with_llm(concept)
                                if parent_concept and parent_concept.lower() != "none":
                                    session.write_transaction(
                                        create_ontology_relationship,
                                        concept,
                                        parent_concept
                                    )

            except Exception as e:
                print(f"❌ Failed processing table {table_name}: {e}")

            # Try to infer relationships between tables with LLM
        try:
            table_nodes = session.run("MATCH (t:DataObject) RETURN t.name AS name, t.schema AS schema").data()
            for i in range(len(table_nodes)):
                for j in range(i + 1, len(table_nodes)):
                    t1 = table_nodes[i]
                    t2 = table_nodes[j]

                    cols1 = session.run("MATCH (t:DataObject {name: $name, schema: $schema})-[:HAS_COLUMN]->(c:Column) RETURN c.name AS name", name=t1['name'], schema=t1['schema']).data()
                    cols2 = session.run("MATCH (t:DataObject {name: $name, schema: $schema})-[:HAS_COLUMN]->(c:Column) RETURN c.name AS name", name=t2['name'], schema=t2['schema']).data()

                    columns1 = [c['name'] for c in cols1]
                    columns2 = [c['name'] for c in cols2]

                    fk_prompt = foreign_key_prompt.format(
                        table1_name=f"{t1['schema']}.{t1['name']}",
                        table1_columns=columns1,
                        table2_name=f"{t2['schema']}.{t2['name']}",
                        table2_columns=columns2
                    )

                    response = llm.invoke(fk_prompt).strip()
                    if "->" in response:
                        left, right = response.split("->")
                        left = left.strip()
                        right = right.strip()
                        session.run("""
                            MATCH (c1:Column {qualifiedName: $left}), (c2:Column {qualifiedName: $right})
                            MERGE (c1)-[:POSSIBLE_FOREIGN_KEY]->(c2)
                        """, left=left, right=right)
        except Exception as fk_e:
            print(f"⚠️ Foreign key inference failed: {fk_e}")

    driver.close()
    conn.close()
    print("✅ Metadata and ontologies loaded into Neo4j.")

def classify_column_with_llm(col_name, dtype, table, schema, description, sample_data):
    formatted_prompt = prompt.format(col_name=col_name, dtype=dtype, table=table, schema=schema, description=description, sample_data=sample_data)
    return llm.invoke(formatted_prompt).strip()

def get_parent_concept_with_llm(concept):
    prompt_text = hierarchy_prompt.format(concept=concept)
    return llm.invoke(prompt_text).strip()

def create_dataobject(tx, name, schema, source, datasource_id, table_id):
    tx.run("""
        MERGE (t:DataObject {name: $name, schema: $schema, sourceSystem: $source})
        SET t.datasourceId = $datasource_id, t.tableId = $table_id
    """, name=name, schema=schema, source=source, datasource_id=datasource_id, table_id=table_id)

def create_column_and_relationship(tx, table_name, schema_name, source_system,
                                    col_name, data_type, is_nullable, comment, table_id, concept, embedding, concept_embedding):
    qualified_name = f"{schema_name}.{table_name}.{col_name}"
    tx.run("""
        MERGE (c:Column {qualifiedName: $qname})
        SET c.name = $name,
            c.dataType = $dtype,
            c.isNullable = $nullable,
            c.description = $comment,
            c.embedding = $embedding
        WITH c
        MATCH (t:DataObject {name: $table_name, schema: $schema, sourceSystem: $source})
        MERGE (t)-[:HAS_COLUMN]->(c)
        MERGE (concept:Concept {name: $concept})
        ON CREATE SET concept.embedding = $concept_embedding
        MERGE (c)-[:HAS_CONCEPT]->(concept)
    """, qname=qualified_name, name=col_name, dtype=data_type, nullable=is_nullable, comment=comment,
         table_name=table_name, schema=schema_name, source=source_system, concept=concept, embedding=embedding, concept_embedding=concept_embedding)

def create_ontology_relationship(tx, child, parent):
    tx.run("""
        MATCH (child:Concept {name: $child})
    MATCH (parent:Concept {name: $parent})
        MERGE (child)-[:IS_A]->(parent)
    """, child=child, parent=parent)

if __name__ == "__main__":
    load_metadata_into_neo4j()
