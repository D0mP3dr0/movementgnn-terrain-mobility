# Provenance

This file lists, for every file in the repository, where it comes from, the SHA-256 of the file as run or as
written, the SHA-256 of the file published here, and what differs. `SHA256SUMS` carries the published digests
in the format read by `sha256sum -c`.

## Sources

- **frozen source**: the frozen version of the pipeline that produced the reported results (commit
  `e00fa11d8919151a2674e731d4663c0ef4c5e69e`, 2026-07-12). Every Python source of `src/`, `scripts/` and
  `train.py`, the final weights and `results/metrics_final_run.json` come from it.
- **release 1.0.x**: the files of the earlier public release that are not part of the frozen pipeline: the
  figure scripts and previews, the manifests, the documents, the workbooks and the audit records.

## What differs from the files as run or as written

- **Python sources.** Comments and docstrings were removed; string literals of four or more words (console
  messages, help texts) were translated into English. The absolute data-root prefix hard-coded in the scripts
  (a Google Drive folder on Colab or a local drive folder) was replaced by the variable `INTERNAL_DATA_ROOT`, read
  from the environment variable `MOVEMENTGNN_DATA_ROOT` in one statement added at the top of each such script; in
  one console hint that prefix became the generic folder name `internal`. Thirteen short console strings remain
  in Portuguese. `verify_code.py` recomputes the digest of each syntax tree with those parts normalized and
  compares it with the digest of the file as run, listed in `CODE_AST_SHA256`.
- **`src/outputs/.gitkeep`.** `src/__init__.py` imports `src.outputs`, an empty folder in the pipeline as run;
  git does not store empty folders, so this empty file keeps it.
- **YAML configuration.** Comments were translated into English and one comment quoting a local folder was
  removed; `verify_code.py` checks the parsed content.
- **JSON and CSV records.** Free-text notes and CSV headers were translated into English; the absolute data-root
  prefix of the quoted paths became `${MOVEMENTGNN_DATA_ROOT}` and the home folder of a local Python installation
  became `${USER_HOME}`; the keys of the four audit records t2, t4, t4b and t12 were translated (the map back is
  in `verify_payload.py`). `verify_payload.py` recomputes the payload digest of each record (structure, keys
  as written, values, and the numbers inside each translated note) and compares it with `PAYLOAD_SHA256`.
- **Workbooks (`.xlsx`).** The last-modified-by field of the document properties held a personal name and was
  cleared; every other part of each workbook is byte-identical, and the digest of all cell values (sheet,
  coordinate, type, value) is the same before and after, as given in the last column. Some cell notes are in
  Portuguese and were kept.
- **`docs/Graph_Construction.md`.** Translated into English; its numbers are unchanged.
- **`models/movement_gnn_best.pt`.** Byte-identical to the weights as run; the training configuration stored
  inside the checkpoint quotes the absolute data paths of the training machine.

Files of release 1.0.x that are not in this release: `src/paths.py` (part of the restructured copy of the
pipeline, not imported by the code as run), the full-resolution `figures/preview/fig01_study_area.png`, the
supplementary information PDF and the supplementary spatial-files archive (each larger than 5 MB; the spatial
files are archived under DOI 10.5281/zenodo.21329327).

Because the bytes of the scripts changed, a digest of a script recorded elsewhere refers to the file as run;
the third column gives those.

## Every file

| file | source | sha256 as run or as written | sha256 published here | check |
| --- | --- | --- | --- | --- |
| `.gitignore` | new | n/a | `1d1e7142c3570b4b930f9631974a9e97f70c47d9d8ab12de8dab49131f77f17b` | written for this release |
| `CITATION.cff` | new | n/a | `42c7a78bf9b4780f73d518cc18393ea4a12c060ae51a6810f90975fc22352ce1` | written for this release |
| `LICENSE` | release 1.0.x | `89c3189f3a789adaaa6f67dcc536c92b632a5131ef62bff0b8cc09801ff73b30` | `89c3189f3a789adaaa6f67dcc536c92b632a5131ef62bff0b8cc09801ff73b30` | byte-identical |
| `LICENSE-DATA` | new | n/a | `abc36908567eda20ea203073b92981c8368f047f6ffc7fe9a9e1dfdd2a33cff8` | written for this release |
| `README.md` | new | n/a | `2f4141b2460aec659481921a41b8be89c5b25aa097a345ec80be5f9691c7523d` | written for this release |
| `configs/config_movementgnn_v2.yaml` | release 1.0.x | `c90d09c25c99a487e9621c59f639a3bf71de90b831994d496e7c754055cdab40` | `5518f003ad44da42b6836e8aa39584f220650c30a31104d42ffad98f982f15d4` | code digest `716eff340a976b24…` |
| `data/Source_Manifest.csv` | release 1.0.x | `2dd87041b596036af86f6ee6a97278d503489acd018601e5aef68e7762cb1170` | `e13f572d8d88d6d662540b944e73eb0e4ee748c13430765e83b4f9dd52ec86bc` | payload digest `c7f786e6271d9dd4…` |
| `data/split_indices.npz` | release 1.0.x | `13dfb586dc1afe2976d568f1f52d19a573370c7ffb477dd285b8280aea1e7ab6` | `13dfb586dc1afe2976d568f1f52d19a573370c7ffb477dd285b8280aea1e7ab6` | byte-identical |
| `docs/Feature_Dictionary.xlsx` | release 1.0.x | `e646908f1264f61d8a7376d918d0497a489290870bd30ea8a876de155e62c0a7` | `750117821816dadabb34eaf39edeb46002836f20a10867d8dc4777bd79edb4a6` | document metadata cleared; cell digest `532cf28e62357881…` unchanged |
| `docs/Graph_Construction.md` | release 1.0.x | `379b9e051c262fcf41c29a081a9e1307671163e81446e9211cb9e66d0c322c29` | `7893f00b9cf1a03e78e5ee40a55b4b268e66117a2629450fa72d8ddbda84fdc5` | translated into English |
| `docs/Label_Generation_Rules.xlsx` | release 1.0.x | `7a7228db81b39aa6094ea50addad6211c58a73e3fa7a18092300cad66c3f22b0` | `1fdacd39efd7b7ecfdfc65679e637f97cb3efd3c18cada8540efd05aeb1e7a37` | document metadata cleared; cell digest `e86e6fadf8299161…` unchanged |
| `docs/Model_and_Training_Config.xlsx` | release 1.0.x | `aaccb293f92452f90a66e401eb52197c06062abb93baf223ed53a2c0fc5c855f` | `25d8eaac96f169c28d20277fe510937783803a1a3ee7f402632710e8b6f588f9` | document metadata cleared; cell digest `fb0dff4047351c2c…` unchanged |
| `docs/Spatial_Split_Manifest.csv` | release 1.0.x | `4f2291f9475acde08ef4532c57fcfe017d4a672697ff8d70dd2dc76094db1f0b` | `2dab4d38da69d1491db7043f201c9954aba5fad8dc859371f6c415f0f653597d` | payload digest `c25251b56a3da64c…` |
| `figures/Figures_Manifest.csv` | release 1.0.x | `413c52b4dec9547130ddb77fcb2efb0ad147497430f9b6c2a77de6073ad93b15` | `8f2ba3dec95505951a8e38dbdbb65e47199f510259f396b10e1cce1b56064b00` | payload digest `1e51c80d476cf759…` |
| `figures/build_fig_spatial_split_v2.py` | release 1.0.x | `e0659f2f3397095467cb8c8330c42228fb34a4213167fc3e1014ff4599596e92` | `3bf0f54ba95862b127e53029c318d09fff43bc960b1a121ee67a054fd5a9e1e6` | code digest `f65c8b52e40b3796…` |
| `figures/fig01_study_area.py` | release 1.0.x | `4a36f9a4686d2ddcd55cee7cb58e4011cd5c90ecb5b0e0d6e3a41a45a4496729` | `c032861a0505fcd04e9c719e27a9b7192e0537476c55b61789b77fcb6f4f48b5` | code digest `6e4ae14a6698c75e…` |
| `figures/fig02_pipeline.py` | release 1.0.x | `ff8dcf6586d0d1dea22a46f981da991d399c43543f2c093f2c8ecc62c738071d` | `c60570fa3432b7cc0369d3460921c49d9646e291304ea746deb92af292460159` | code digest `835e93c1e66f814c…` |
| `figures/fig03_architecture.py` | release 1.0.x | `1d8afd8a1d15d415f0164843ddcde55cce7f89504562858990f710ec0e2ef892` | `0070efb851ee99d1cf0a197bf6de4e77a6d0caec547449a6663b92494308d1b3` | code digest `31d87ffce94bfeb7…` |
| `figures/fig04_comparative.py` | release 1.0.x | `9a5f3b964a7863e02737954574c2ffebd65626f9dbd68d6c5476fba9ee35a8e9` | `3609e42d3e171d5da2d107e650c763cadea4a5bd8f4202b1631a334a868f65f9` | code digest `26a28f66f9a93dcf…` |
| `figures/fig05_intelligence_gap.py` | release 1.0.x | `f47f6129260a174e33bd2351cfd7faa1907b26e4b6f3263e9326708614b6b33e` | `2c7d72109e7cd8103e12ace4118bd343b2d4ac96cdeeb4244bfca5ae87945997` | code digest `0ca6a70ca38ad052…` |
| `figures/fig06_performance.py` | release 1.0.x | `229052a0c414e3dfa632dbbe5448b10b9bd43e8395bf2e59acb966c8f5aa533f` | `7c19caa17447b6b0a289c2e5c8ec724cc4f24e903cc91d1851fa789df78a6e68` | code digest `fa968e6c7e1de0c7…` |
| `figures/fig07_asymmetric_error.py` | release 1.0.x | `5c38226b256e551ad879938b2e04f676760e23f4003bbc442ecca824e406b68a` | `946768324015986968921ab8cab169da6529dae4f0b7d623a83e6101aa267519` | code digest `c95b671b8f67b050…` |
| `figures/fig_correlation.py` | release 1.0.x | `c741fc1e85acf60872f63a1e9047799ac1035c9ba09f61d0e6da91034ceed69f` | `84cb430b9a2814c4aa4ef81bc04e3be7cfeeb4fa80a032266b1f783e1e7f5e85` | code digest `9d43c8a9aed02aa7…` |
| `figures/preview/fig02_pipeline.png` | release 1.0.x | `dca3b67c45312506bcbb3b2a0e704684b7c4813fef8918feea4a461c7b5dd0a5` | `dca3b67c45312506bcbb3b2a0e704684b7c4813fef8918feea4a461c7b5dd0a5` | byte-identical |
| `figures/preview/fig03_architecture.png` | release 1.0.x | `48335d33bf2d199bf21b4c68b1b24ae60da06e7db72e4354b808d6b4c3283c6e` | `48335d33bf2d199bf21b4c68b1b24ae60da06e7db72e4354b808d6b4c3283c6e` | byte-identical |
| `figures/preview/fig04a_terrain.png` | release 1.0.x | `406ee41af89777c47edbd1313ab31a200f87849c951ce83da5bca0427ed599dd` | `406ee41af89777c47edbd1313ab31a200f87849c951ce83da5bca0427ed599dd` | byte-identical |
| `figures/preview/fig04b_dismounted.png` | release 1.0.x | `52b615a6176ea2d6a8ac4a2aa555cfe7ece843804fce657403450146070cfd41` | `52b615a6176ea2d6a8ac4a2aa555cfe7ece843804fce657403450146070cfd41` | byte-identical |
| `figures/preview/fig04c_armored.png` | release 1.0.x | `399c59ac8cdbb163faa5c478004cf6833520225edead8c041824aba59d3a06f6` | `399c59ac8cdbb163faa5c478004cf6833520225edead8c041824aba59d3a06f6` | byte-identical |
| `figures/preview/fig04d_motorized.png` | release 1.0.x | `3f0dd07802395bbd06bcccf82483929f19074830f3761e24c7a3ff342080180d` | `3f0dd07802395bbd06bcccf82483929f19074830f3761e24c7a3ff342080180d` | byte-identical |
| `figures/preview/fig05_intelligence_gap.png` | release 1.0.x | `8256b7f2c37afd2aa5500bd8f4cf2a46291cd54d0ff2c502b84e05099e7191b2` | `8256b7f2c37afd2aa5500bd8f4cf2a46291cd54d0ff2c502b84e05099e7191b2` | byte-identical |
| `figures/preview/fig06_performance.png` | release 1.0.x | `3749483a6466e41952d1a1b1182b6f8e65854dd0b300c76e76a24da887d80f01` | `3749483a6466e41952d1a1b1182b6f8e65854dd0b300c76e76a24da887d80f01` | byte-identical |
| `figures/preview/fig07_asymmetric_error.png` | release 1.0.x | `1093b28e751259a02cf5e44b374bc8c8d913a8cfec83e95b65fbf1fee57b5bc7` | `1093b28e751259a02cf5e44b374bc8c8d913a8cfec83e95b65fbf1fee57b5bc7` | byte-identical |
| `figures/preview/fig_correlation.png` | release 1.0.x | `2fffd2a8eb9a6858740d30cfea716effc9293df1b47d7e3bc183a6a6969b0b55` | `2fffd2a8eb9a6858740d30cfea716effc9293df1b47d7e3bc183a6a6969b0b55` | byte-identical |
| `figures/preview/fig_spatial_split_map.png` | release 1.0.x | `765bc38648bfa0df9e97866e8f3467d09c5c2c12b23e6623d4e341c3d44b3d55` | `765bc38648bfa0df9e97866e8f3467d09c5c2c12b23e6623d4e341c3d44b3d55` | byte-identical |
| `figures/utils_nato.py` | release 1.0.x | `d9c62893540a8fac3f361734cd42ddc7619f4a44c0c2f875256b61abf4f2daf9` | `7c46ca0201e5678f125f25e55eddaa2b0c4cf84cb3b26b833a0b9c0f5eb677d6` | code digest `b6e29a31b3b3f67e…` |
| `models/movement_gnn_best.pt` | frozen source | `85f21530cdbca6caefcb457083afa249eb9bb07f3e1f5d1ffae5b739d51711f7` | `85f21530cdbca6caefcb457083afa249eb9bb07f3e1f5d1ffae5b739d51711f7` | byte-identical |
| `requirements.txt` | new | n/a | `6978b4493a25dc12feddd5b3ebd75ecde9fedb73031a421c9c4d11a448fefe39` | written for this release |
| `requirements_freeze_full.txt` | release 1.0.x | `4f91cb7b64c81676d7e7c03e18b13e0d037bc491f6e6adfbd9a0a7beeed3cdb8` | `4f91cb7b64c81676d7e7c03e18b13e0d037bc491f6e6adfbd9a0a7beeed3cdb8` | byte-identical |
| `results/Results_Master.xlsx` | release 1.0.x | `8f31f397dfdbf3a83fde8df5b394210e612c28047863b7aba621ef6bd5cd4b9f` | `8f31f397dfdbf3a83fde8df5b394210e612c28047863b7aba621ef6bd5cd4b9f` | byte-identical |
| `results/metrics_final_run.json` | frozen source | `9468bfe7619a48f8d4bab673cf319c4c90613e1a8f44ee5854f1607c904e55e0` | `66ca22ef439148785e7aea44d74445daad117336e11d353898823972cea5c644` | payload digest `6846c1aee776ec69…` |
| `results/t12_rebuild_timing.json` | release 1.0.x | `21b56d4f52ea4dea6f10ad619b6b0cc4db09351b9a5126eba7d79cc7867196dd` | `c766c36238784e21097acc2f662da6049f1ce0debbbbf96bf777e91a68e900aa` | payload digest `8ad06f6f3ef29e50…` |
| `results/t2_threshold_bands.json` | release 1.0.x | `85a209722d993b385353039ae770c94bf72ca834716a7f31f4db0289cf14735c` | `c22b9373d0e7da70e3db587e126391e89eb7c87faf1f16ab410129b01d1b3c35` | payload digest `080f2a7ba9cc3cc3…` |
| `results/t4_audit_results.json` | release 1.0.x | `1489033c08704f5b68c724945f728ab70e1507cbaacbe31582a6fefbba382c73` | `59234d7fca6851090366b6101086fe4199fc573bac76d28b1d9bc8d2cea85119` | payload digest `2625a894eca45305…` |
| `results/t4b_block_structure.json` | release 1.0.x | `8f00ed76ccacc3be334e6ab8f8cc0583bcb4b4b81b6d890f8d14f6657d1cd3b6` | `6fef5137dddad110382377a93791551e8cb3f469089c926462e6fe6565b02e6b` | payload digest `6afad413e56494e0…` |
| `scripts/analysis/09_diagnostico_transicao.py` | frozen source | `feeda5160f11e2a964112e4437dc8331b101630cc5959572643c7cad5830d5de` | `ab2898426808ea637aff45897f515a3b914f10833b5f92a275d9cf2562087b00` | code digest `2e5b7d575cf380a8…` |
| `scripts/analysis/10_analise_incerteza.py` | frozen source | `58665ecf0d8641b246d5ce8af40fb1323480ad5c20e94f964f48768b48aaa76e` | `a962a24be3d0ca8dac435d8618bd982c0b9049fd8546de79a61b9fd31827f018` | code digest `50f9d2d1992a1430…` |
| `scripts/analysis/13_label_audit.py` | frozen source | `1529b1710e85c438491e437d793fa4051a929947af0c36eed690e2f7945b8004` | `1f6b793cabb993ac45a9cd15f4419eb5c37ac200ccbe001fa9ac2ecabd5463fa` | code digest `8dc383d4888b42b6…` |
| `scripts/analysis/14_embeddings_analysis.py` | frozen source | `b335f9fb6c1e3408cc380d8dad2a07287e569ba73c0138dc9e77fdb8506d97e4` | `5a476f07a9304b6912feaecf95b20e383b0647795c7256c6b4b34779b0ec4c35` | code digest `8e2dc03b94a15178…` |
| `scripts/analysis/15_analise_geoespacial.py` | frozen source | `9ca9e60cbabec457418433fb674b1d75f967a52c3d3f3f0c5e6fcbc46117afa6` | `82430f65fbbf93b38f5c6b13124b7126d0cc032cb3397d4939478e53cc6d3c7f` | code digest `f8b097f2e34219a3…` |
| `scripts/analysis/analyze_dataset.py` | frozen source | `6a13f3990b980daac61d59e39d25ca1d8a8b56c59a77c2ab1db98288da98319d` | `70a2b807e64b0ef717aef2aa9b44bcdd709236d84804fcc9147f7b7c238751d4` | code digest `603c58379dfbe733…` |
| `scripts/data/06b_extract_topo_embeddings.py` | frozen source | `1da71fa7355636839dccddbc841358badb1da895ffd0520e0f9e8d112e78d81d` | `57e5971548e77c5abe170ca73d48815bc824fd14e287296a3f8320313056d319` | code digest `f02b3e3fc179d1c7…` |
| `scripts/data/06b_extract_topo_embeddings_local.py` | frozen source | `6de19289e44eec36dc7d2adf83a3b0ae3ac8e214fad21d4e1615501cecbf5c6a` | `4edda7a998cf75a5354160c9945864edb74d3139c166407036311992d7a5db41` | code digest `6034ec5cd4754eb4…` |
| `scripts/data/download_dem_pacaraima_q1q4.py` | frozen source | `da6be054f5ab34befdb7227d3801a7e9843968dc67870853c050828570b24f67` | `898835ea8c4dfa3627f6dc97bed3d40912d2b0a04dd8d803792fc38d4287dbc7` | code digest `c324ef1482f4be86…` |
| `scripts/data/download_lidar_pacaraima_q1q4.py` | frozen source | `d3d5eb5e568dd8c2e6d851a4b7c3f0bda2d70df347747fc4a17e57a51a4621e4` | `1e6d36eb7880916245690f53f91a6495307c4154978640d58db805091008eeff` | code digest `498d97b25cd9d49e…` |
| `scripts/data/download_pacaraima_full_package.py` | frozen source | `25abf2e2329ecf0b10203ed15e313c78f7d03bff3a47df49ef7391b652649b49` | `7a612816033350c4d23cc14ffe66546060899a0906b0e17bc145c30bc45d77cb` | code digest `343f038fc8c5912a…` |
| `scripts/data/download_sentinel2_pacaraima_q1q4.py` | frozen source | `2e0e3714dd776daf081b13fe17599212e0d5f8944f1ba6854fe0543fe6bc2618` | `79462861160e11a7f27f377f4332b41c8fb4a3a805ea93c6224e8c8afd772ed4` | code digest `0334bb2727e47132…` |
| `scripts/inference/11_gerar_probs.py` | frozen source | `69d01e13d93253c7e2ab535686be731875bcb8a9efa2c94ebb94fd9e9351b4a6` | `2900f5dc4bd5e06f3ec1ae32c1f561ba85b39fbf85beb4ba539361a2abaee993` | code digest `747abff262b0e8ed…` |
| `scripts/inference/12_gerar_zrn.py` | frozen source | `643c0f978eb9888c040c668e03810045e8dbed75234c9aa580f2884298fa8e90` | `7c43f83b372def0cf220a0753f2b2c9d91d1d7ccb7bda3459e2dfa388019f455` | code digest `318517478c04ef2a…` |
| `scripts/inference/re_export_geotiff.py` | frozen source | `239f3a6841053e7e00c3019fd0fe949bd88bcddd2298f79904065737afcfae71` | `afd103800f867c7e3452b8acaf288fa32fc50ce59a55c94c25874e891c58748b` | code digest `c7696c1076ab5fd4…` |
| `scripts/training/07_train_gnn.py` | frozen source | `451853d680154032cfa3200af3e7b4739b6a5719f7e210a2b46da3c6afd907d2` | `1f59c244314787d47839dd464951f96aca84f2fb0171e35dae2889d193c6c8cb` | code digest `a68bd370fd023550…` |
| `scripts/training/07_train_gnn_v2.py` | frozen source | `9b763935bd9f287de3d94b6ceca782bd7a77644508915ecb53a6c64691a8ef75` | `4f027f59130ce2152ecb5f7502cc50d2611f79ece7165860112ecd77a59a66f0` | code digest `8801ab320dfac932…` |
| `scripts/training/07b_retrain_blindada_fix.py` | frozen source | `c8dd405addce04a9a2b4df3318ad4acba1484fb7a0375e21a721c924325e138a` | `5be380e2a9aee1d0baca87ef689406ad69418240dc5903be964af1cace7797b5` | code digest `47972895a12ddf3e…` |
| `scripts/training/08_resume_train.py` | frozen source | `cf8990a7a3374141dbc27c23bccce8e477ae81b1515e405187310490335d5db9` | `bc18ba9bbf62c4e4123ac5463dd8bb03f75e03aba70b93ed39e7dc700c0e1a04` | code digest `b57dc1e6025a2b8b…` |
| `src/__init__.py` | frozen source | `372be09fc75800627a8f1818abf79df3be9d068215b1c24cbc109b858f526cab` | `af832252518355e8c2c58b2e895dacb0f7eb04520520accc3e2a77bc16fbfe4a` | code digest `454595e4d78c3f87…` |
| `src/baselines/__init__.py` | frozen source | `752e67c118ebee2855df7c70d9c09da2506c6b177619472c2a02681352b5da81` | `013f1d75b7c7bcd6f925efa7a2fdcccc7358ed639c5907c4a30f8d235b463d49` | code digest `73465048453e1b7e…` |
| `src/baselines/baseline_runner.py` | frozen source | `3a90bdb7b44b41ce0d5ff8241107130cf06bf455f1c57ecedc0c88461458bd5e` | `1f1b2196202a2497b3dd0565a1c09fbcfe8cc48f78d0e9019001097fd93cc959` | code digest `50e142d0acf8bb7a…` |
| `src/baselines/export_baseline_to_geotiff.py` | frozen source | `a3260960e14dbbf345a304635d15315db04c9eaeafa9556c6e679c36e492bf79` | `0fa9694c57f10afef2b708bc1a4dbe3a68e69e230ae6a51f3819320196bd6e53` | code digest `7c79d1798da4d69f…` |
| `src/baselines/mlp_baseline.py` | frozen source | `6d19581d5f33fd11b5181be266379ac98658107b7079ae1698342019b7d06d0f` | `3a5b39fe4a9a1575cf64c74b0e14f563631a57076fb550a1127cda20f6c71fb9` | code digest `cf727a9e9bf2be62…` |
| `src/baselines/random_forest_baseline.py` | frozen source | `2f1d1558b4e3d49960840fd2bf9c5bcfce6d8e62466d0bc15c96f090b74eee71` | `9f130c8933dc85ff8175343a8475c8ade64757879948cebd9a66b4be3a76065d` | code digest `8da1dc4aa710c9c6…` |
| `src/baselines/rule_based_baseline.py` | frozen source | `21c32b387b7cc7b055095d9b0081a79ab350aa10b0cd779ba477575a5c797637` | `5dc5452abf0c054cc1f5a77da1f04cacca5728ffab54942f98c58d457449da5f` | code digest `d375b0bd7f1aabd3…` |
| `src/data_processing/__init__.py` | frozen source | `f7f4eae2ec1a42f7693d6cde96f8091fefba9ebde7c2c9e2710f4879594f9368` | `ea7b8b266b80381c7ef421180e30df358ddd5a9d5c8f77d557c76cf5f17f7cc9` | code digest `dbfd58e70c5c791f…` |
| `src/data_processing/data_validator.py` | frozen source | `05e0d879dc47c522768285821a1163c80d7e844202048b529ba2411548d2f135` | `c03d0f00af20b31aa2ddd66ba6c6e05a83bfa00c1670e4624419a13ec36757b6` | code digest `1408b64fd2838651…` |
| `src/data_processing/dataset_loader.py` | frozen source | `08ac5fe08c6e63f3f8728006aabed008382d36f78c75073a4fca60ded7a69fe9` | `67cdc5eb6036851875bc422ae42c45a58afefdc779e9bad1b244ad6b0a376bff` | code digest `66f5f05234184e27…` |
| `src/data_processing/feature_extractor.py` | frozen source | `d5cc328489f952b1f3f3b3f2827e75cd9ec6da56751c6892b89429614a6c3284` | `8d2069d380514b8c44380b3c8b8587cef8ae022d033ffa65e54d5b307448e922` | code digest `0ad1c8f9855ca428…` |
| `src/graph_construction/__init__.py` | frozen source | `1534d62e1f340a804fa6ee45bba41331e4ced3e0d11f7e828177cdd6ae26c040` | `35be36b1b5c87c5e9efa3820c40d41cce40cac3bcc226487d6aca6fa99b3994b` | code digest `5d087e7ea74e3677…` |
| `src/graph_construction/graph_integrator.py` | frozen source | `543eb7aa22c1509f3971fde68f29a237e3ab4dff27158cfd7df239218ae50bc8` | `2dc91ea81c628dbf8ab1d5ca94630972a315ab48829bbd42edcc9d62882f4f62` | code digest `299eb7a3bc7820a8…` |
| `src/graph_construction/spatial_splitter.py` | frozen source | `4b4c7cdf97e34b3659e8af8a7e0cbb541592c8aad83a06a223a014bc8f7a2b25` | `bcca933d044feb35524b426d948f4b86e41f4f1b64b7d284966af6fca0da6939` | code digest `307dc376aa441043…` |
| `src/label_generation/__init__.py` | frozen source | `c5b319b096ea47bb29255c7deb198bbd1d974508b4cde019bfdff25b1c5318f8` | `d2df208086fef2e9910780f9f5f18fd1db9cb26c2912c401cedf0baf9667c972` | code digest `61aa82c30fc974b9…` |
| `src/label_generation/dameplan_rules.py` | frozen source | `0c139f77edfc53e9737b2fe81d71710ed526381627b3b4b9463dea0f0c308852` | `d81b52f34c43396c24c43f9a05adc83d986a06618cc510d707b19dc55248a7c2` | code digest `628bb3ff646e5048…` |
| `src/label_generation/label_generator.py` | frozen source | `a90aaaf77b577d5f1c7554cecc224287d2c0e083388e2adfe41107605adeae69` | `27c786c531f02a96f2e4b70f2c4559097e0dd9a6205467fdca251acc25364117` | code digest `fcce8b4b5a41d725…` |
| `src/label_generation/label_validator.py` | frozen source | `ce75df3de1ca055a73008bceeb60c01aa433153c43210cce10c5405c642e2d3b` | `70d4e03c5d66468a208e9f3c9d9bb4b9ae533cacb4d0adce9021b32ba7bb4870` | code digest `d947c8d01fb5553a…` |
| `src/models/__init__.py` | frozen source | `a5be9068efcc523a8b72c98e380349a075f7fafb8a0eb3c37c033fde6036c9fe` | `d010f6d001834b1a5b224a8a57fe38582e2cb7b5d92cd0e576c00942048d47c5` | code digest `133da5dd2247eadd…` |
| `src/models/dameplan_loss.py` | frozen source | `0a1e054a0608d12e58d4c51e30a7acbe41ba2a38554dd813aee89311bf97a7da` | `d463a077c0b1d9a3e235032415283306ea9ed3357f2a4ac7aa7504feeb3284c3` | code digest `4d21ac878e3f950d…` |
| `src/models/focal_loss.py` | frozen source | `71f474339f18b7d465b4c862bd16c5874c46d9c9d7d8ca2432e76119be53b15a` | `04a1a2ca815afd8e9d361abebc62422e790c879f22732ae8217a3a2df21caeca` | code digest `6e9e01ae1d453feb…` |
| `src/models/movement_gnn.py` | frozen source | `4680e13151e885f382af08a2c70552043912cc39b6608c7b8f2455d51545eb20` | `78a96c40667af9991e85628c35424729d6bf62051196592eae998924b0944240` | code digest `068039508d689781…` |
| `src/outputs/.gitkeep` | new | n/a | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | written for this release |
| `src/training/__init__.py` | frozen source | `a87b60d199bed55a2f0607615b9c122d74e4644ac673da869db7c312daa3b8ee` | `eac68399cb6637a6d6cf51ef7fdb7656014d3dc4ac5d64008e0e715943421e05` | code digest `202d6c57ffe650e0…` |
| `src/training/early_stopping.py` | frozen source | `3fee77792011376b7d5dccde5a4bb9e1ecb561318fe64678fc68e0e44746d5c2` | `958e1009d0acfa6857706816ced34c919acc7096d45d7a0bd5e029d9092a9b53` | code digest `9b04706bac911e9b…` |
| `src/training/trainer.py` | frozen source | `1000586daf76402055ac85f090a97171c426bade02b8cd530f1f73b7f0724af0` | `ad94eabdba8071543b4707b60aaa95e9c744b2313e6a41948a1b5f2bceb27157` | code digest `511309c83f7251a1…` |
| `supplementary/Supplementary_Tables.xlsx` | release 1.0.x | `95ae3b979e52f556de4a42859b501c0d273f7c0bddc4a85073dc2ecb2734ba34` | `749f528b78383e60247498eb2982c4bb7fbd37f094e72e2a8497f694396b8395` | document metadata cleared; cell digest `abf7002eafe99ef2…` unchanged |
| `train.py` | frozen source | `e9ecc5d65c21033846014cedf46b161bfc520d1c3e565ff31974d7805680caa4` | `8d8f384765b74d54d59efd043585dc932c8e1340e476939cf1a135637f37fb13` | code digest `01719c9993027925…` |
| `verify_code.py` | new | n/a | `e651cd994dfd312677314800931523864fdbbe425b25b39f05f9c61e0a6d8b14` | written for this release |
| `verify_payload.py` | new | n/a | `b27ce15cce6b02252ab52cad1ab88f87394594c2f9db6c0c37beaa80adce48b4` | written for this release |
