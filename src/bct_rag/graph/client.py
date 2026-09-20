import os

from neo4j import GraphDatabase


class GraphClient:
    """
    Thin wrapper around the Neo4j driver.

    Connection defaults target a local Neo4j instance (bolt://localhost:7687),
    matching how you'll run scripts directly on your host. Inside the
    rag-builder Docker container, NEO4J_URI is set to bolt://neo4j:7687
    (see docker-compose.yml) so the same code works in both places.
    """

    def __init__(self):

        uri = os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        )

        user = os.getenv(
            "NEO4J_USER",
            "neo4j",
        )

        password = os.getenv(
            "NEO4J_PASSWORD",
            "password",
        )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )

    def close(self):
        self.driver.close()

    def execute(self, query: str, parameters: dict | None = None):

        with self.driver.session() as session:

            result = session.run(
                query,
                parameters or {},
            )

            return list(result)