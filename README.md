# U-Net: Convolutional Networks for Biomedical Image Segmentation

This repository contains code for replicating the results of the paper titled "U-Net: Convolutional Networks for Biomedical Image Segmentation." ([Link to Paper](https://arxiv.org/abs/1505.04597))

## Introduction

Biomedical image segmentation plays a crucial role in various medical applications, such as disease diagnosis and treatment planning. The U-Net architecture is a widely recognized deep learning model for biomedical image segmentation. This repository provides the code necessary to replicate the results and explore the U-Net architecture for biomedical image segmentation.

## Requirements
You can install required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Training and Evaluation
1. **Training**
- Refer to the `Usage.ipynb` notebook for a step-by-step guide on how to train the model.
- You can customize training configurations, such as batch size, learning rate, and more, within the notebook.
- Experimentation: If you want to experiment with different configurations, you can modify the settings directly in the notebook during training.

2. **Evaluation**
- Use the `Usage.ipynb` notebook to visualize and evaluate the results.
- You can customize evaluation options, within the notebook.

By following the instructions in the `Usage.ipynb` notebook, you can easily replicate the experiments, visualize the results, and conduct further experiments with different settings.

For more detailed instructions and examples, please refer to the notebook itself.

## Citation
If you use this code or replicate the results, please consider citing the original paper:
```bash
@article{arXiv:1505.04597,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2015}
}
```
## Acknowledgements
We extend our thanks to the authors of the original paper, "U-Net: Convolutional Networks for Biomedical Image Segmentation," for their valuable research and inspiration.

## References
- [Implementing original U-Net from scratch using PyTorch](https://www.youtube.com/watch?v=u1loyDCoGbE)
- [PyTorch Image Segmentation Tutorial with U-NET: everything from scratch baby](https://www.youtube.com/watch?v=IHq1t7NxS8k&t=1378s)

## License
This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
