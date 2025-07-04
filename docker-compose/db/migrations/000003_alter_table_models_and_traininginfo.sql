BEGIN;

ALTER TABLE models ADD COLUMN case_id VARCHAR(36);

ALTER TABLE models 
ADD CONSTRAINT fk_models_case_id 
FOREIGN KEY (case_id)
REFERENCES cases(case_id);

ALTER TABLE training_info ADD COLUMN case_id VARCHAR(36);

ALTER TABLE training_info 
ADD CONSTRAINT fk_training_info_case_id 
FOREIGN KEY (case_id)
REFERENCES cases(case_id);

COMMIT;