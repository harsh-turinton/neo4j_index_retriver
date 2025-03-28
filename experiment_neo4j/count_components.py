from neo4j import GraphDatabase

# --- CONFIG ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "harsh@123"

def count_graph_components():
    """
    Counts the number of connected components in the Neo4j graph.
    Attempts to use APOC plugin first, then falls back to plain Cypher.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("Counting graph components...")

        print("Using basic Cypher query (this might take a while for large graphs)...")
        try:
            result = session.run("""
                MATCH (n)
                WITH id(n) AS nodeId
                WITH collect(nodeId) AS allNodes
                WITH allNodes, [] AS components
                // This is a simplified approach that works for small graphs
                UNWIND allNodes AS nodeId
                MATCH (n) WHERE id(n) = nodeId
                MATCH path = (n)-[*0..1]-(m)
                WITH collect(distinct id(m)) AS component
                RETURN COUNT(DISTINCT component) AS componentCount
            """)
            count = result.single()["componentCount"]
            print(f"✅ Number of connected components in the graph: {count}")
            return count
        except Exception as e:
            print(f"❌ All methods failed: {e}")
            return None
    
    driver.close()

def print_node_counts():
    """Print the count of different types of nodes in the graph"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("\n--- Node counts by type ---")
        result = session.run("""
            MATCH (n)
            WITH labels(n) AS nodeType, count(n) AS count
            RETURN nodeType, count
            ORDER BY count DESC
        """)
        
        for record in result:
            node_type = ':'.join(record["nodeType"])
            count = record["count"]
            print(f"{node_type}: {count} nodes")
    
    driver.close()

def print_relationship_counts():
    """Print the count of different types of relationships in the graph"""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        print("\n--- Relationship counts by type ---")
        result = session.run("""
            MATCH ()-[r]->()
            WITH type(r) AS relType, count(r) AS count
            RETURN relType, count
            ORDER BY count DESC
        """)
        
        for record in result:
            rel_type = record["relType"]
            count = record["count"]
            print(f"{rel_type}: {count} relationships")
    
    driver.close()

if __name__ == "__main__":
    # Count graph components
    count_graph_components()
    
    # Print additional information about the graph
    print_node_counts()
    print_relationship_counts()
    
    print("\nDone!")