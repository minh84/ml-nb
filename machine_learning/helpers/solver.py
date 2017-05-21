from .funcs import get_batch_indices

def solve(model, optimizer, thetas_0,
          train_X, train_y, val_X, val_y,
          epochs, batch_size, debug = False):

    optimizer.minimize(model)
    model.init_thetas(thetas_0)
    dbg_loss = None

    if debug:
        dbg_loss = []
    N = train_X.shape[0]

    for e in range(epochs):
        for idx in get_batch_indices(N, batch_size):
            batch_X = train_X[idx]
            batch_y = train_y[idx]

            # run optimizer update
            optimizer.step(batch_X, batch_y)
            # store error if debug = True
            if debug:
                train_err = model.error(batch_X, batch_y)
                valid_err = model.error(val_X, val_y)
                dbg_loss.append([train_err, valid_err])

    return dbg_loss