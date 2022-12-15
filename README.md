# TreEnhance
Official implementation of the paper "TreEnhance:  A Tree Search Method For Low-Light Image Enhancement"

[Marco Cotogni](https://scholar.google.com/citations?user=8PUz5lAAAAAJ&hl=it) and [Claudio Cusano](https://scholar.google.com/citations?hl=it&user=lhZpU_8AAAAJ&view_op=list_works&sortby=pubdate)

Accepted in Pattern Recognition

[Paper arXiv](https://arxiv.org/pdf/2205.12639.pdf)

<p align="center">
<img src="figs/tree.png" width="400" height="350"/>
<img src="figs/opt.png" width="400" height="150" />
</p>

## Requirements
python > 3.7, Pytorch, Torchvision, PIL, numpy

## Running Experiments

To reproduce the experiments of our paper, please download the [LOL](https://daooshee.github.io/BMVC2018website/) and [FIVE-K](https://data.csail.mit.edu/graphics/fivek/)
Split the data in train and test folders and run the training.py script. Once the model has been trained, it could be tested using the evaluation.py

## Results
<p float="left">
  <img src="figs/grid.png" width="350" height="400" />
  <img src="figs/lol.png" width="350" height="400" />
</p>

## Reference
If you are considering using our code or you want to cite our paper please use:

```
@article{cotogni2022treenhance,
  title={TreEnhance: An Automatic Tree-Search Based Method for Low-Light Image Enhancement},
  author={Cotogni, Marco and Cusano, Claudio},
  journal={arXiv preprint arXiv:2205.12639},
  year={2022}
}
```

