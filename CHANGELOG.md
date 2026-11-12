# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/IGNF/ssl-models/compare/v1.3.0...HEAD)</small>

### Features

- add `average_precision` and `average_recall` for `segment` and `change_detect` targets.
- log per-class IoU, F1, precision and recall, enabled with `run.log_per_class_metrics` (options: `per_class_metrics`, `per_class_stages`).
- add optional `class_names` to target configs, set for FLAIR (cosia), Hi-UCD, PASTIS-HD, S2-NAIP and Tempetes.
- label confusion matrix axes with class names, save one `.npy` per epoch with a `_classes.json` file, and allow disabling it with `run.log_confusion_matrix`.

<!-- insertion marker -->
## [v1.3.0](https://github.com/IGNF/ssl-models/releases/tag/v1.3.0) - 2025-12-04

<small>[Compare with v1.2.0](https://github.com/IGNF/ssl-models/compare/v1.2.0...v1.3.0)</small>

## [v1.2.0](https://github.com/IGNF/ssl-models/releases/tag/v1.2.0) - 2025-11-05

<small>[Compare with v1.0.0](https://github.com/IGNF/ssl-models/compare/v1.0.0...v1.2.0)</small>

### Features

- move target configs in init (#73) ([88153a4](https://github.com/IGNF/ssl-models/commit/88153a460646d08d7642b9d304dfafe722c1dc64) by Antoine Labatie).

### Bug Fixes

- fix computation of logits in probing (#72) ([e23e8d8](https://github.com/IGNF/ssl-models/commit/e23e8d89ac99d291c8d50d7b0cb18fac6010bb4c) by Antoine Labatie).

## [v1.0.0](https://github.com/IGNF/ssl-models/releases/tag/v1.0.0) - 2025-10-20

<small>[Compare with first commit](https://github.com/IGNF/ssl-models/compare/541fdc3af49b1306736e7e44899514dad18213cb...v1.0.0)</small>

### Features

- add bdf-v2 dataset (#60) ([fbf7a54](https://github.com/IGNF/ssl-models/commit/fbf7a547e2eba3b04f0b2d3ac2cf2b684f42b9a8) by Antoine Labatie).
- allow loading of PyTorch model from config and weights path (#59) ([6dfc854](https://github.com/IGNF/ssl-models/commit/6dfc85419eccb8245a00b7d2c3a1c75bbb448fdc) by Antoine Labatie).
- add use_ema argument in BaselineModel configuration dataclass ([5d46874](https://github.com/IGNF/ssl-models/commit/5d46874813373de1a24610252cb3d52ad4063964) by Michael Vaccaro).
- add date encodings and s1_asc s1_desc support ([69b82a7](https://github.com/IGNF/ssl-models/commit/69b82a7ccd43c80467a368f7c826398fc21a627e) by Michael Vaccaro).
- paper default configs (#45) ([7e848f6](https://github.com/IGNF/ssl-models/commit/7e848f63daa0c21f3ff9eb2b2028b651487db4f7) by Antoine Labatie).
- make date encodings addition optional (part2) ([8e53f0a](https://github.com/IGNF/ssl-models/commit/8e53f0aa93e4d133810a205bf754724cb2bf8b71) by Michael Vaccaro).
- make date encodings addition optional ([33fa616](https://github.com/IGNF/ssl-models/commit/33fa6160e9c9abcfe8b51a7683ddc0843e595f83) by Michael Vaccaro).
- patch_size for baseline models ([a7281f0](https://github.com/IGNF/ssl-models/commit/a7281f0eb15ecad16405f4588ec6a1fb89fc3a8e) by Michael Vaccaro).
- croma file creation ([ab7f5a6](https://github.com/IGNF/ssl-models/commit/ab7f5a6110b0c3d11f767a79863af86b39636c25) by Michael Vaccaro).
- possibility to disable the transfer of the last layernorm layer ([7567f36](https://github.com/IGNF/ssl-models/commit/7567f364561f715dad5e0ac1c9f0ee722877c551) by Michael Vaccaro).
- add multi encoder support ([62ce7e5](https://github.com/IGNF/ssl-models/commit/62ce7e58a4d7b656bd5400256eeb42a015303747) by Michael Vaccaro).
- add DOFA support ([f08230c](https://github.com/IGNF/ssl-models/commit/f08230ca36d590bb9d691274bbbec3e43dbd3f82) by Michael Vaccaro).
- layerwise lr decay during finetuning ([5805f52](https://github.com/IGNF/ssl-models/commit/5805f52f04d56d4cca91d2bdacbe8574392a66cc) by Michael Vaccaro).
- allow accounting of date axis in loss normalization ([0ef38af](https://github.com/IGNF/ssl-models/commit/0ef38afb22167595e47de1792cee25aab5cf62ed) by Antoine Labatie).
- permutation of s2 rgb channels ([68f3be4](https://github.com/IGNF/ssl-models/commit/68f3be4cc097817d708f4d48b054181cc73972b7) by Michael Vaccaro).
- multi patch embedding ([ca38367](https://github.com/IGNF/ssl-models/commit/ca383678789509ae7b798f92ed0e2ba6ff751502) by Michael Vaccaro).
- add flair version ([88df364](https://github.com/IGNF/ssl-models/commit/88df3643dd3e8a8414390f28f310f01625073227) by nlardiere).
- add dinov2 additive date encodings ([8a70af8](https://github.com/IGNF/ssl-models/commit/8a70af8abfb2cda8dd5ad8c438e2386e16684e17) by Michael Vaccaro).
- add tiny randomly initialised model option for more efficient debugging ([58b2505](https://github.com/IGNF/ssl-models/commit/58b25053bbcb0ca9183d61777fdbd16fb08db84f) by Michael Vaccaro).
- plot confusion matrix in tensorboard images ([95e713d](https://github.com/IGNF/ssl-models/commit/95e713dd31fd07a6b10f39c2fa0fe69dbe8ea678) by Antoine Labatie).
- adapt dinov2 to multimodal multitemporal data (shared weights) ([1414979](https://github.com/IGNF/ssl-models/commit/141497917d02f6053b9af203e40dc982acaf157a) by Michael Vaccaro).
- add EMA + final div factor argument for optimizer ([38744e0](https://github.com/IGNF/ssl-models/commit/38744e02fe563c46392e012a9a9cb71330b79835) by Antoine Labatie).
- add confusion matrix ([eede143](https://github.com/IGNF/ssl-models/commit/eede1433203a10ae39c882938952777587a793a4) by nlardiere).
- allow tiling of val/test zones ([fbaedbf](https://github.com/IGNF/ssl-models/commit/fbaedbf2328984d82b0669dd173395985446593c) by Antoine Labatie).
- add weight in loss + allmods in decoder ([fb3e44f](https://github.com/IGNF/ssl-models/commit/fb3e44ffe234315e45113835639ec13e6f876e3a) by Antoine Labatie).
- change defaults FLAIR-INC ([fc6d85b](https://github.com/IGNF/ssl-models/commit/fc6d85bbdf8facb0834d5a3532ea98255f2517e7) by Antoine Labatie).
- add repeats arguments for segmentation/change datasets ([9dbffac](https://github.com/IGNF/ssl-models/commit/9dbffac2905aedd3936bf0fc471c19362da58971) by Antoine Labatie).
- move batch size option to optimizers ([e66c7dc](https://github.com/IGNF/ssl-models/commit/e66c7dce13f39fb815a83cce44d841e851597555) by Antoine Labatie).
- change defaults ([84644df](https://github.com/IGNF/ssl-models/commit/84644dfd3d596f3a46c2548ba516ea98e0918a11) by Antoine Labatie).
- add option to monitor metric in probing/finetuning ([cc433ba](https://github.com/IGNF/ssl-models/commit/cc433ba9a56e0372722594627c8ebeb520ddd8d2) by Antoine Labatie).
- add options for train+val SSL pretraining + probing/finetuning on filtered dataset ([6ae7876](https://github.com/IGNF/ssl-models/commit/6ae78765291ead3043eb41e2e22c2e456ed70722) by Antoine Labatie).
- add norm band + batch repeat options ([f92a288](https://github.com/IGNF/ssl-models/commit/f92a2887ec0042cba930eeefc2581c4c528f0eac) by Antoine Labatie).
- allow using balanced dataset for pretraining ([7bfc107](https://github.com/IGNF/ssl-models/commit/7bfc1076babd21ab7c711e110a66563550f94ff7) by Antoine Labatie).
- improve unpooling ([790bb2c](https://github.com/IGNF/ssl-models/commit/790bb2c54103468cc3aa0d08cf36e13c2f6895d0) by Antoine Labatie).
- remove date patching ([48db612](https://github.com/IGNF/ssl-models/commit/48db61293a06cd0881d290e715e4f84d0a49ecad) by Antoine Labatie).
- allow disabling of scaling of mask ratio ([33a6700](https://github.com/IGNF/ssl-models/commit/33a67008a47ac1e302aee704ac1dee2ffc955836) by Antoine Labatie).
- allow random dates selected uniformly on patch size ([5109259](https://github.com/IGNF/ssl-models/commit/5109259c1b207cb9c155428760009526f4e4ff93) by Antoine Labatie).
- allow combination of raster and classif targets ([30c73cb](https://github.com/IGNF/ssl-models/commit/30c73cb5ef37eb3fb192f859f1b1db7b9057e83c) by Antoine Labatie).
- add layer-wise decay ([73cda7e](https://github.com/IGNF/ssl-models/commit/73cda7e0512b14312a7c9ad97a163097dcf6bef0) by Antoine Labatie).
- allow different shapes in grouped modalities ([fba3e51](https://github.com/IGNF/ssl-models/commit/fba3e51df4ef858b5e8c4c3b96deba6ee20ba4c7) by Antoine Labatie).
- working version of structural masking ([457ad72](https://github.com/IGNF/ssl-models/commit/457ad72ddaacb3888ff42f451fec342f57d71f66) by Antoine Labatie).
- add token split by band groups and grouping of modalities ([8652ae3](https://github.com/IGNF/ssl-models/commit/8652ae3ee4c1a7408572e4c5f74d24ea0b1049c0) by Antoine Labatie).
- allow unpool with patch size not a power of 2 ([5a3cb13](https://github.com/IGNF/ssl-models/commit/5a3cb139afac40ea20fded1283925a915ad19126) by Antoine Labatie).
- add flair dataloader ([ab47b1e](https://github.com/IGNF/ssl-models/commit/ab47b1efde58eb4bf54177c3be081825ad82355b) by nlardiere).
- make mask ratio dependent on number of dates with dependent masking ([269a45b](https://github.com/IGNF/ssl-models/commit/269a45b6be14c7d889628979d140a65d3100b1d6) by Antoine Labatie).
- add test set ([327e3fe](https://github.com/IGNF/ssl-models/commit/327e3fe95933afe3f63820425a96ea1df1f07697) by Antoine Labatie).
- rework multimodal models ([5f6af85](https://github.com/IGNF/ssl-models/commit/5f6af85cf0994bc11c8d2f9a4a595da8619b9a7a) by Antoine Labatie).
- config pastis dataset ([a95d30f](https://github.com/IGNF/ssl-models/commit/a95d30f486fc819f2d707230e610dfb7622bc581) by nlardiere).
- add pastis dataset ([85fc5d2](https://github.com/IGNF/ssl-models/commit/85fc5d2acf2348500a350cfa08fbcb23c820808d) by nlardiere).
- allow multimodal MAE ([97624d2](https://github.com/IGNF/ssl-models/commit/97624d2dad498c0452cd6e4e83b2975303f65e44) by Antoine Labatie).
- finalize inclusion of configs for new datasets ([62ff5c8](https://github.com/IGNF/ssl-models/commit/62ff5c8dc8516cd8e7adeff8d454cff61cb18fad) by Antoine Labatie).
- working version with new heads and metrics ([8a3786e](https://github.com/IGNF/ssl-models/commit/8a3786e157d5b6664e522f3201bdf5d50a2b2fd5) by Antoine Labatie).
- first working version with TreeSatAI-TS ([ef30703](https://github.com/IGNF/ssl-models/commit/ef30703d424d118e41e1488dc03e088ada23beb8) by Antoine Labatie).
- add cropping and GSD ([d98d4b5](https://github.com/IGNF/ssl-models/commit/d98d4b540825940204baa567402abd0bc53622cc) by Antoine Labatie).
- add TreeSatAI-TS dataset ([4ecbf7a](https://github.com/IGNF/ssl-models/commit/4ecbf7a2db06d3b5a93da321c435bd5bf4eabd0b) by Antoine Labatie).
- refactor configs from main function ([48fa685](https://github.com/IGNF/ssl-models/commit/48fa6850e32d8035c6cb00f5c43d5e75102d257d) by Antoine Labatie).
- add dataset configs ([5d00f35](https://github.com/IGNF/ssl-models/commit/5d00f358a5a1b05ea6f584cf183fa8a01527d7d3) by Antoine Labatie).
- reintroduce basic lightning data module ([c550f30](https://github.com/IGNF/ssl-models/commit/c550f3022156002eebbae5eadc6f84ae846da7f2) by Antoine Labatie).
- allow reloading of pretrained model ([d417785](https://github.com/IGNF/ssl-models/commit/d4177851bcde8e2d54ef4d6165ab3a029e9b3ebd) by Antoine Labatie).
- add Swin improvements ([01d369f](https://github.com/IGNF/ssl-models/commit/01d369f413323ead753663b7e7b5178dc23484de) by Antoine Labatie).
- add classification metrics ([03c02f0](https://github.com/IGNF/ssl-models/commit/03c02f0de5dc2ed4f702568bc08adfacd2bd4251) by Antoine Labatie).
- add swin version as parameter ([29d1df3](https://github.com/IGNF/ssl-models/commit/29d1df34feba35df1e00fe4bdea8ab7c2be30d7b) by nlardiere).
- switch to hydra configs ([a302f94](https://github.com/IGNF/ssl-models/commit/a302f947e9baeaf560409ca29787a428dcfca7ac) by Antoine Labatie).
- mix 4 images ([eb9d0d0](https://github.com/IGNF/ssl-models/commit/eb9d0d0f70362844d5de03323035fb5af9cf1da9) by nlardiere).
- add probing ([cbd8155](https://github.com/IGNF/ssl-models/commit/cbd8155cacf0fc4ee930fe45e5c230013fd092b5) by Antoine Labatie).
- add ViT-T and ViT-S and remove patch normalization before loss ([608fd61](https://github.com/IGNF/ssl-models/commit/608fd61dd45df5e27c6cb6bf1554e50485af551f) by Antoine Labatie).
- add unpooling option in patchify/pixelify ([89018db](https://github.com/IGNF/ssl-models/commit/89018db6ffd4d40b48d84fa81a2a5b95ff5ca821) by Antoine Labatie).
- add support for ClearML ([bdff6ed](https://github.com/IGNF/ssl-models/commit/bdff6ed0758cfce7631b20d37c34a26359c95594) by Antoine Labatie).
- make image logger compatible with distributed training ([de64139](https://github.com/IGNF/ssl-models/commit/de6413910f01f5c71a95fccbb99d7d797b26f403) by Antoine Labatie).
- improve learning rate scaling and scheduler ([1c77445](https://github.com/IGNF/ssl-models/commit/1c77445b65f24c5cdecccf9214eff73b15ce0233) by Antoine Labatie).
- add option to specify whether val dataset is shuffled ([f9a4396](https://github.com/IGNF/ssl-models/commit/f9a4396e0b3504febb9229df34d2ba82b09fe215) by Antoine Labatie).
- improve image logging ([d8c2228](https://github.com/IGNF/ssl-models/commit/d8c2228212ce1eb0342ef7a644548b712eb1f8b3) by Antoine Labatie).
- first working version ([c9c25a2](https://github.com/IGNF/ssl-models/commit/c9c25a272ce61baf45e2f14844acea38083c9a09) by Antoine Labatie).
- run toy pretext task ([9e12360](https://github.com/IGNF/ssl-models/commit/9e12360e7c318086cb893ac4b8339ec39fb1c067) by Antoine Labatie).
- add MAE based on clay foundation model code ([f176779](https://github.com/IGNF/ssl-models/commit/f176779f4e1054150306bbd668f9c8d60699b454) by Antoine Labatie).
- switch to ruff linter and formatter ([64a6448](https://github.com/IGNF/ssl-models/commit/64a644812404dad579c0ac1d359c00f2822a5a56) by Antoine Labatie).
- initialize repo ([5f02f1a](https://github.com/IGNF/ssl-models/commit/5f02f1aa778951370b26b3025e64fc5b989ccb3f) by Antoine Labatie).

### Bug Fixes

- add date encoding croma ([a73656f](https://github.com/IGNF/ssl-models/commit/a73656fbc8ba1e86cf238fb1b17341b74ef1d8d6) by Michael Vaccaro).
- expand 10-bands sentinel-2 time series to 12 channels (croma) ([8a5bc7a](https://github.com/IGNF/ssl-models/commit/8a5bc7af5e054f1da7df18a78167281e4a449fa4) by Michael Vaccaro).
- merge from master ([c74e624](https://github.com/IGNF/ssl-models/commit/c74e624bfd715abbefd905881c2c446117eaa51d) by Michael Vaccaro).
- dofa._build_multi_encoder does not exist anymore ([9a87d7b](https://github.com/IGNF/ssl-models/commit/9a87d7b598bcb4f9c8d466c158cfe50ca8fb4f10) by Michael Vaccaro).
- multi encoder layerwise learning rate (dofa) ([a67bb6c](https://github.com/IGNF/ssl-models/commit/a67bb6c5d538501d971e2f3fb746f061b4122f70) by Michael Vaccaro).
- learning rate adaptation multi patch embed layer ([aae14fe](https://github.com/IGNF/ssl-models/commit/aae14fe81149a45ec5b3a93cca0bbe2f5dd29633) by Michael Vaccaro).
- layerwise for dofa ([430f5d5](https://github.com/IGNF/ssl-models/commit/430f5d538a4b1ab8a75e0c24045dbbc6a2675b0c) by Michael Vaccaro).
- replace nanmean ([87a77d3](https://github.com/IGNF/ssl-models/commit/87a77d301026788db623032205fb5e56df08487a) by nlardiere).
- transfer weights to dofa backbone ([0fb6478](https://github.com/IGNF/ssl-models/commit/0fb6478efe00858e1a371646e820eb7a6b7f44d0) by Michael Vaccaro).
- metrics monitoring due to failed merge with main, refactor: base class for baseline models ([7fa89f8](https://github.com/IGNF/ssl-models/commit/7fa89f8efdb7d216a956cb78bb6e42be9e248298) by Michael Vaccaro).
- set load_state_dict.strict to False ([c6c06a8](https://github.com/IGNF/ssl-models/commit/c6c06a81da7ce47c28285093757ceddc16b920aa) by Michael Vaccaro).
- renaming state dict keys for transfer learning ([7d9a8d5](https://github.com/IGNF/ssl-models/commit/7d9a8d5144942efccd6f67093111d52121dd7151) by Michael Vaccaro).
- remove nrows in read_csv ([31d36c1](https://github.com/IGNF/ssl-models/commit/31d36c1c442165a57aced7fcedc916d24040dd60) by nlardiere).
- fix rank zero issue ([55560d8](https://github.com/IGNF/ssl-models/commit/55560d814a72e4189df2a37df6af2269a87bcfa8) by Antoine Labatie).
- resize input image before applying segmentation mask ([c9491d4](https://github.com/IGNF/ssl-models/commit/c9491d47ad2d36c5cc2ffa2b11e2f7dedb3ab8d8) by Antoine Labatie).
- downgrade albumentations ([2dcd5a1](https://github.com/IGNF/ssl-models/commit/2dcd5a10230d3b63f9ca8a3fad0120656284b6af) by Antoine Labatie).
- rgb bands order for s2 datasets ([53fe0ba](https://github.com/IGNF/ssl-models/commit/53fe0ba4b4da522f2c1c75fb0b3536a14e233955) by Michael Vaccaro).
- fix uniform cropping in val/test + add folds for PASTIS-HD ([adf8284](https://github.com/IGNF/ssl-models/commit/adf828417d761b5e8b78b2b7eb90ad9fa9777d4a) by Antoine Labatie).
- fix FLAIR data loader ([4c7dd68](https://github.com/IGNF/ssl-models/commit/4c7dd68d24d686d0d8e8ce33473bade1c5e07cfa) by Antoine Labatie).
- fix bug with band groups ([c65d046](https://github.com/IGNF/ssl-models/commit/c65d046ced73013069d2b9afb4bd0cfa2c8683b8) by Antoine Labatie).
- fix bug in structural masking and date with band groups ([14d11c8](https://github.com/IGNF/ssl-models/commit/14d11c875861713b29b2939e5ef5de363c655a8b) by Antoine Labatie).
- fix bug with random dates in train ([a1f6509](https://github.com/IGNF/ssl-models/commit/a1f6509ed3ecde37ba41bfc380325d351e886e57) by Antoine Labatie).
- fix random dates ([3ccadc5](https://github.com/IGNF/ssl-models/commit/3ccadc5c964c2fba3a90029268cc9ace08bf944d) by Antoine Labatie).
- fix MixMAE ([3beba0e](https://github.com/IGNF/ssl-models/commit/3beba0e36458ef741d4b0cada71fc00f5046c1f9) by Antoine Labatie).
- fix dates in PASTIS-HD ([c8c603e](https://github.com/IGNF/ssl-models/commit/c8c603e98ab35326115fe9bb7f85d6b89bd623ab) by Antoine Labatie).
- consider background ([a24d316](https://github.com/IGNF/ssl-models/commit/a24d316690cbf6ea26c307a120afd8042526077e) by nlardiere).
- fix transform by using numpy instead of albumentations ([668b619](https://github.com/IGNF/ssl-models/commit/668b619a81fb2e35ecd7cb347b2d17f08cb1fba0) by Antoine Labatie).
- remove unecessary argument ([12ea5e0](https://github.com/IGNF/ssl-models/commit/12ea5e01d629cd4d361ab37b25da8fc3c0cac75f) by nlardiere).
- remove unecessary argument mod ([c1df702](https://github.com/IGNF/ssl-models/commit/c1df7025626af1433a3fe5b2dfc1d2081ae9cc7c) by nlardiere).
- fix tests ([1999af8](https://github.com/IGNF/ssl-models/commit/1999af8b9d17b74a3091c0cae3fbe83fcccd78e6) by Antoine Labatie).
- fix issues with sorting and torch types ([a8afab5](https://github.com/IGNF/ssl-models/commit/a8afab5c270d25d38a7f27352b5f124099249660) by Antoine Labatie).
- add parameters in test ([1338aca](https://github.com/IGNF/ssl-models/commit/1338acaf30cbe84f9aee6f22cba006ccd4d4f765) by nlardiere).
- add parameter for mixing image number ([1a99d11](https://github.com/IGNF/ssl-models/commit/1a99d115999730b2ca7977323c39f1dcf168424c) by nlardiere).
- remove unused arg ([794fab4](https://github.com/IGNF/ssl-models/commit/794fab4aa78d6e9b98b3abf1206399e4cffca06a) by nlardiere).
- consider all images ([e3be97b](https://github.com/IGNF/ssl-models/commit/e3be97bf25270ef633b93b5925e1ed6b0829dc7d) by nlardiere).
- remove mix images in crossentropoy ([d495fce](https://github.com/IGNF/ssl-models/commit/d495fce02cc570e835f902bebbd0c29caef36f96) by nlardiere).
- remove mixing images for probing/finetuning ([f074719](https://github.com/IGNF/ssl-models/commit/f074719fe92ff56db00cd4d2633828a288161a42) by nlardiere).
- add unpoolig and downsampling arguments for MAE tests ([b9ca95a](https://github.com/IGNF/ssl-models/commit/b9ca95a44123c3dde7905e20980e8db6ac250396) by nlardiere).
- fix bug with batches without labels in validation during probing/finetuning ([5c2b31b](https://github.com/IGNF/ssl-models/commit/5c2b31b3466ba39caf249db614549c135c8440c9) by Antoine Labatie).
- fix accuracy bug ([686b529](https://github.com/IGNF/ssl-models/commit/686b529cc5abf871ac3cceb5822295a1ead8f773) by Antoine Labatie).
- add drop_last to dataloader ([6ee89c2](https://github.com/IGNF/ssl-models/commit/6ee89c2ff0b51fe7eaa6f6fcfedd0d00a845edd8) by nlardiere).
- fix visualization ([e3af2c5](https://github.com/IGNF/ssl-models/commit/e3af2c507aeef82e7f28e917100f232d98c23423) by Antoine Labatie).

### Reverts

- revert direct push ([de7fa2a](https://github.com/IGNF/ssl-models/commit/de7fa2a2ca22a806419d667eb5b69f1082c88ea6) by Antoine Labatie).

### Code Refactoring

- move DEM/elev rescaling to PyTorch module (#63) ([f29bf34](https://github.com/IGNF/ssl-models/commit/f29bf34ee601f350a5cb77467e752a854f10df35) by Antoine Labatie).
- probe_or_finetune_step in base class ([9576168](https://github.com/IGNF/ssl-models/commit/9576168075474361940436e1e8b3dfc50b237a4a) by Michael Vaccaro).
- remove obsolete DOFA class ([e58414e](https://github.com/IGNF/ssl-models/commit/e58414e69852ffff4c9c7906f7d049c3f5776ecb) by Michael Vaccaro).
- build backbone hidden method ([e62a084](https://github.com/IGNF/ssl-models/commit/e62a084527ef7aee33b92bba1b32bf7de9eea6bf) by Michael Vaccaro).
- transfer attributes to BaseModule mother class ([d9797e8](https://github.com/IGNF/ssl-models/commit/d9797e8a742c9e37a9b149f4d32b995b18516378) by Michael Vaccaro).
- channels selection now in dataloader ([0948475](https://github.com/IGNF/ssl-models/commit/0948475247614c2c449dbc30299accd15cf297a0) by Michael Vaccaro).
- simplification of dinov2 weights loading ([e3db8a7](https://github.com/IGNF/ssl-models/commit/e3db8a7d163ae5e9665f38da42caa9e2a49b2fa5) by Michael Vaccaro).
- remove prints ([ba2bb92](https://github.com/IGNF/ssl-models/commit/ba2bb92b8239df579564276ba135796707c2f81b) by Michael Vaccaro).
- abc abstract method for base lightning module ([d78e256](https://github.com/IGNF/ssl-models/commit/d78e256bccd3aae7f27403bd343b97d8be3d4f22) by Michael Vaccaro).
- draft base lightning module ([a0af451](https://github.com/IGNF/ssl-models/commit/a0af4511af8f0a7383867513b4dac699193d9dbe) by Michael Vaccaro).
- refactor generic dataset for efficiency ([bd6e128](https://github.com/IGNF/ssl-models/commit/bd6e12856e01bce1d3475d057d2027438cedc013) by Antoine Labatie).
- refactor file names for lightning data modules ([e9e8270](https://github.com/IGNF/ssl-models/commit/e9e827004f2050c36d06a40e25785cae615f1465) by Antoine Labatie).
- move trainer, model and logger modules in separate train folder ([da72855](https://github.com/IGNF/ssl-models/commit/da72855b5a9c8aa5d3e476957a960ea3af268043) by Antoine Labatie).
- add subfolders for data, layers, and ssl modules ([a10a474](https://github.com/IGNF/ssl-models/commit/a10a474abd97b9ac3f3c7834f7d4d1a41a4d0971) by Antoine Labatie).
- major refactoring with sharing of common MiM module for MAE and MixMAE ([604aaff](https://github.com/IGNF/ssl-models/commit/604aaff7be5b9b35488ac91bc2691fe147bc13af) by Antoine Labatie).

### Performance Improvements

- optimize running time by running decoder/loss only in pretrain, and running heads only in probe/fine-tune ([7fab3b7](https://github.com/IGNF/ssl-models/commit/7fab3b7097b4737a39099bd550ceb11f609ad4a2) by Antoine Labatie).

