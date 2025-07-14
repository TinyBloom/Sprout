BEGIN;

CREATE TABLE hyper_parameters (
    param_id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(48) DEFAULT 'IsolationForest',
    param_key VARCHAR(48),
    description TEXT,
);

CREATE TABLE hyper_parameter_values (
    value_id VARCHAR(36) PRIMARY KEY,
    param_id VARCHAR(36) REFERENCES hyper_parameters(param_id) ON DELETE CASCADE,
    param_value VARCHAR(255) NOT NULL
);

COMMIT;