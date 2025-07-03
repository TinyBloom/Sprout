BEGIN;

ALTER TABLE models ADD COLUMN case_id VARCHAR(36);

ALTER TABLE models 
ADD CONSTRAINT fk_models_case_id 
FOREIGN KEY (case_id)
REFERENCES cases(case_id);

COMMIT;