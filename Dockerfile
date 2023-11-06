FROM neo4j:5.13-community
WORKDIR /var/lib/neo4j
EXPOSE 7474
EXPOSE 7687
VOLUME $HOME/nlp_etl_demo/container/data
ENV NEO4J_AUTH=none
