# To do

## Optimisation step (29/10)
- [ ] Text: remove the 50k feature limit (`max_features=None`). CV shows +1.2 points.
- [ ] Tune with cross-validation on train only. Keep validation for the final comparison.

## Mentor feedback still open
- [ ] 1. Same validation set for text and image (no rows removed from validation)
- [ ] 2. Image notebook: fix keys with spaces, run top to bottom
- [ ] 3. Image model: explain train loss > val loss; cache frozen ResNet features
- [ ] 6. Per-class F1, text vs image side by side
- [ ] 7. Fusion: calibrate SVM scores into probabilities

## Done
- [x] 4. Feature limit checked, benchmark note added
- [x] 5. Seed check for binary TF-IDF
- [x] 8. One category-name table, small text fixes
