docker run \
-p 7474:7474 \
-p 7687:7687 \
--restart unless-stopped \
--detach nlp-etl-demo
