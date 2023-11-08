import neo4j
from langchain.vectorstores import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from os import environ
import json


class Neo4jResults:
    def __init__(self, data=None, metadata=None):
        self.data = data or []
        self.metadata = metadata

    @classmethod
    def from_result(cls, result: neo4j.Result):
        return cls(data=result.data(), metadata=result.consume())

    def __call__(self):
        return self.data

    def __repr__(self):
        return json.dumps(self.data)


class Neo4JDatabase:
    def __init__(
        self,
        protocol="neo4j",
        hostname="localhost",
        port=7687,
        username="neo4j",
        password="OpenSaysMe",
    ):
        self.protocol = protocol
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password

    @property
    def url(self):
        return f"{self.protocol}://{self.hostname}:{self.port}"

    def __enter__(self):
        self.driver = neo4j.GraphDatabase.driver(
            self.url, auth=(self.username, self.password)
        )
        return self.driver.session()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.close()

    @classmethod
    def from_environment(cls):
        return cls(
            protocol="neo4j",
            hostname=environ["NEO4J_HOST"],
            port=environ["NEO4J_BOLT_PORT"],
            username=environ["NEO4J_USERNAME"],
            password=environ["NEO4J_PASSWORD"],
        )

    def fetch_labels_and_properties(self):
        with self as session:
            query = "MATCH (n) RETURN DISTINCT labels(n) as labels, keys(n) as propkeys"
            r = session.execute_read(self.transaction, query)
            return r

    @staticmethod
    def transaction(txn, query, **kwargs):
        result = txn.run(query, **kwargs)
        return Neo4jResults.from_result(result)

    def make_vectorstore(
        self,
        embedding,
        index_name="id",
        node_label="Chunk",
        text_node_properties=["id", "text", "title", "source", "embedding"],
        embedding_node_property="embedding",
    ) -> Neo4jVector:
        return Neo4jVector.from_existing_graph(
            embedding=embedding,
            url=self.url,
            username=self.username,
            password=self.password,
            index_name=index_name,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
        )
