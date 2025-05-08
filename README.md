Example inverse rendering processes with outputs are included in the following two notebooks:
```
banana_example_view11_14views.ipynb
banana_example_view11_6views_vis14views.ipynb
```
The first notebook performs inverse rendering using all 14 views.\
The second notebook performs inverse rendering using only 6 views, but the reconstructed model is visualized from all 14 views.

Example reconstructed models (`ply` and `obj` files) can be found in the `example_reconstructed_models/` directory

To run the inverse rendering shape optimization example:
1. Install Dependencies
```
pip install matplotlib
pip install mitsuba
pip install cholespy
```
2. Run `banana_optimization.ipynb`
