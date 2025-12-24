# Surrogate Environment – Model Family

This folder contains MG surrogate environments and their
corresponding user response models.

## Versioning Rule

env_mg_surrogate_vX uses user_response_fitting_vX by default.

## Model Evolution

### User response fitting 
v0: (Complaint-based)
- User response target: complaint人数
- Load fitting: RandomForestRegressor
- Complaint fitting:
  - Classifier (是否投诉)
  - Regressor (投诉人数)
v0_0: origin 
v0_1: bug-fix

v1: (Cost-based)
- User response target: cost
- Load fitting: RandomForestRegressor
- Cost fitting:
  - RandomForestRegressor

v1_0 : basic 
v1_1 : - is_user_dynamic= True
v1_2 :(inherit v1_0): base-complaint-unit-cost: scale to 0.0001 
v1_3 :(inherit v1_0): peak-valley price policy

⚠️ v1 changes the modeling target and is NOT backward compatible with v0.



