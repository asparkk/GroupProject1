DATA PREPARATION

WHAT THIS DOES
prepare_data(folder, seed) reads the MNIST files, makes the pixel numbers
small, sets aside some pictures for checking, and gives back six arrays.

WHAT YOU GET BACK
Name    |  Shape               | Type |   What it is 
--------|----------------------|------|------------------------|
X_train |  shape (48000, 784)  | float|  pictures for training
y_train |  shape (48000,)      | int  |  answers (0-9) for training
X_val   |  shape (12000, 784)  | float|  pictures for checking while we build
y_val   |  shape (12000,)      | int  |  answers (0-9) for checking
X_test  |  shape (10000, 784)  | float|  pictures for the final test
y_test  |  shape (10000,)      | int  |  answers (0-9) for the final test

WHAT I DECIDED AND WHY
- Data: the original MNIST files. 60,000 training pictures and 10,000 test
  pictures. Each picture is 28 x 28.

- Flatten: each picture becomes one row of 784 numbers.

- Scale: every pixel is divided by 255, so it is between 0 and 1.

- Split: the 60,000 training pictures are mixed up with a fixed seed. 48,000
  are for training and 12,000 are for checking.

- Test pictures: I only scale them. I do not mix them or use them to tune
  the model.

- Labels: whole numbers from 0 to 9.

- Mini-batches: make_batches(X, y, batch_size, seed) mixes the data and cuts
  it into small groups. The last group can be smaller.

- Seeds: the same seed always gives the same result. Use a new seed each
  time you go through all the training pics to get a new order.

HOW I CHECKED IT
- count_overlap(A, B) counts pictures that are in both sets. It gave 0 for
  train and validation, train and test, and validation and test.
- check_pipeline.py runs everything and checks sizes, number ranges, overlap
  and repeatability.
- Tests: test_preprocess.py, test_batches.py and test_overlap.py.

HOW TO USE IT
    from preprocess import prepare_data, make_batches
    data = prepare_data("path/to/mnist/folder", 42)
    batches = make_batches(data["X_train"], data["y_train"], 64, 1)
