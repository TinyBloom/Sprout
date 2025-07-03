BEGIN;
CREATE TABLE cases (
    case_id VARCHAR(36) PRIMARY KEY, -- 
    name VARCHAR(255) NOT NULL,
    robot_id VARCHAR(255) NOT NULL,
    case_type VARCHAR(48), -- health_profile, temp_predictor
    model_name VARCHAR(48), -- e.g.: IsolationForest
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
COMMIT;