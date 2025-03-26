
CREATE TABLE models (
    model_id VARCHAR(36) PRIMARY KEY, -- 
    name VARCHAR(255) NOT NULL, -- e.g.: IsolationForest
    robot_id VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE training_info (
    training_id VARCHAR(36) PRIMARY KEY, -- 
    model_id VARCHAR(36) NOT NULL,
    robot_id VARCHAR(255) NOT NULL,
    hyperparameter JSON NOT NULL, -- e.g.: {"n_estimators":"", "contamination":""}
    training_status VARCHAR(48), -- pending, running, completed, failed, canceled
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE model_files (
    file_id VARCHAR(36) PRIMARY KEY, -- 
    model_id  VARCHAR(36) NOT NULL,
    file_name  VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,  -- e.g.: minio://bucket_name/object_name or /local_path
    file_size BIGINT NOT NULL,
    file_format VARCHAR(32) NOT NULL,
    file_hash  VARCHAR(64) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);