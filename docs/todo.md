# To do

## Optimisation step (29/10)
- [ ] Text: remove the 50k feature limit (`max_features=None`). CV shows +1.2 points.
- [ ] Re-test binary TF-IDF with CV on train (it was chosen on validation).
- [ ] Fusion: choose the text/image weight with CV on train, not on validation. Needs image predictions on train.
- [ ] Rule: choose every setting with cross-validation on train. Use validation once, for the final score.

## Mentor feedback still open
- [ ] 1. Same validation set for text and image (no rows removed from validation)
- [ ] 3. Image model: explain train loss > val loss; cache frozen ResNet features

## Done
- [x] 2. Image notebook runs top to bottom
- [x] 4. Feature limit checked, benchmark note added
- [x] 5. Seed check for binary TF-IDF
- [x] 6. Per-class F1, text vs image on the same products
- [x] 7. Fusion first version (local only, weight chosen on validation)
- [x] 8. One category-name table, small text fixes
