#!/bin/sh
set -e

/usr/local/bin/docker-entrypoint.sh postgres &

# Store the PostgreSQL PID
PG_PID=$!

# Wait for PostgreSQL to start
until pg_isready -U sprout_admin -d sprout_model; do
  echo "Waiting for PostgreSQL to start..."
  sleep 1
done

echo "PostgreSQL has started, beginning migration scripts..."

# Execute all SQL files in the /migrations directory in alphabetical order
for sql_file in $(find /migrations -name "*.sql" | sort); do
  echo "Executing migration script: $sql_file"
  psql -v ON_ERROR_STOP=1 -U sprout_admin -d sprout_model -f "$sql_file"
done

echo "All migration scripts executed"

# Keep the container running, handing control back to the PostgreSQL process
wait $PG_PID