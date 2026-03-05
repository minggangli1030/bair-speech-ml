# 6. Failure Case Analysis

We select one correctly classified and one misclassified example from the Control vs. svPPA task (S_ph + S_sem logistic regression, 5-fold CV).


### Case A: Correctly Classified

- **Subject:** sub-1619, **Session:** ses-20171206

- **True label:** control

- **Prediction:** control


**Transcript excerpt:**

> on the lake as a sailboat with people sailing, there's also appears to be someone. itting on a pure fishing. There's also, there also appears to be a child playing in sand. They are in the water, but there's a sand bucket in a shovel. There's a young, maybe a teenager or young person flying a kite. They also appear to be with a dog. And there's a flight pole with a flight flying, so there must be a little bit of a breeze. On the left there's a house. with a car in the driveway. in the foreground...


**Scene graph comparison:**

- Entities mentioned: tree, family, book, dog, house, man, boy, girl, lake, picnic basket, blanket, kite

- Entities missed: sky, boat, drink, woman, grass, food, ball, path, kite string

- Content units covered: ['CU_A01', 'CU_A02', 'CU_A03']

- Content units missed: ['CU_E01', 'CU_E02', 'CU_E03', 'CU_E04', 'CU_E05', 'CU_E06', 'CU_E07', 'CU_E08', 'CU_E09']

- S_sem: 0.379, entity recall: 0.571, CU coverage: 0.250


**Proxy/Adequacy scores:**

- S_ph: 0.422, S_sem: 0.477, Δ: -0.055

- phoneme_error_rate: 0.578, silence_ratio: 0.148


**Clinical (Layer 2) scores:**

- l2_reduced_speech_rate: 0.681

- l2_reduced_length_complexity: 0.435

- l2_agrammatism: 0.244

- l2_paragrammatism: 0.193

- l2_anomia: 0.219

- l2_empty_speech: 0.373

- l2_semantic_paraphasias: 0.289

- l2_phonemic_paraphasias_neologisms: 0.353


### Case B: Misclassified

- **Subject:** sub-10683, **Session:** ses-20181019

- **True label:** control

- **Prediction:** svPPA


**Transcript excerpt:**

> flag is waving, cars part in the driver, you can turn the garage. に Обじってください embarrassing um, some people I have picked neck are under a big tree. There are areas of Bucket and a shovel on the beach. The flag is waving. Radio is playing for the people having a picnic. Man is reading a book. Having the one of the picnickers, the man is reading a book. The shoes are off. Lady is pouring a drink. The... someone on the boat is waving to Pughon, sure. Um, the kite has a


**Scene graph comparison:**

- Entities mentioned: tree, book, family, drink, boat, man, woman, kite

- Entities missed: sky, dog, house, grass, food, girl, boy, ball, lake, path...

- Content units covered: ['CU_A03']

- Content units missed: ['CU_E01', 'CU_E02', 'CU_E03', 'CU_E04', 'CU_E05', 'CU_E06', 'CU_E07', 'CU_E08', 'CU_E09', 'CU_A01', 'CU_A02']

- S_sem: 0.202, entity recall: 0.381, CU coverage: 0.083


**Proxy/Adequacy scores:**

- S_ph: 0.413, S_sem: 0.406, Δ: 0.007

- phoneme_error_rate: 0.587, silence_ratio: 0.140


**Clinical (Layer 2) scores:**

- l2_reduced_speech_rate: 0.611

- l2_reduced_length_complexity: 0.408

- l2_agrammatism: 0.501

- l2_paragrammatism: 0.559

- l2_anomia: 0.649

- l2_empty_speech: 0.289

- l2_semantic_paraphasias: 0.460

- l2_phonemic_paraphasias_neologisms: 0.689


**Interpretation of failure:**

The control was misclassified as svPPA. Possible reasons: (1) S_sem was low (reduced semantic content or scene-graph coverage), driving the model toward svPPA; (2) transcript may be brief or underspecified, resembling svPPA's empty speech pattern; (3) S_ph may be moderate, not providing a strong control signal.
