FROM confluentinc/cp-kafka-connect:8.2.0

# Pin the real Confluent JDBC connector used by the acceptance test. The
# worker image deliberately does not bundle connectors so this stays explicit.
RUN confluent-hub install --no-prompt confluentinc/kafka-connect-jdbc:10.9.5
