"""TensorFlow / Keras feed-forward network (regression: MSE head, classification: sigmoid head).
Early stopping + best-weight restore on validation loss."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from src.models.base import BaseModel

HP = dict(hidden=(64, 32), dropout=0.2, l2=1e-4, lr=1e-3, batch_size=64, max_epochs=300, patience=20)


class _Keras(BaseModel):
    framework = "keras"

    def _build(self, n_in: int):
        from tensorflow import keras
        hp = {**HP, **self.hp}
        reg = keras.regularizers.l2(hp["l2"])
        layers = [keras.layers.Input(shape=(n_in,))]
        for h in hp["hidden"]:
            layers += [keras.layers.Dense(h, activation="relu", kernel_regularizer=reg),
                       keras.layers.Dropout(hp["dropout"])]
        reg_task = self.task == "regression"
        layers.append(keras.layers.Dense(1, activation=None if reg_task else "sigmoid"))
        model = keras.Sequential(layers)
        model.compile(optimizer=keras.optimizers.Adam(hp["lr"]),
                      loss="mse" if reg_task else "binary_crossentropy")
        return model

    def _fit(self, X_train, y_train, X_val, y_val):
        from tensorflow import keras
        hp = {**HP, **self.hp}
        keras.utils.set_random_seed(self.seed)
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        self.model_ = self._build(X_train.shape[1])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=hp["patience"], restore_best_weights=True)
        h = self.model_.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=hp["max_epochs"],
                            batch_size=hp["batch_size"], callbacks=[es], verbose=0, shuffle=True)
        self.epochs_trained_ = len(h.history["loss"])
        self.history_ = {k: [float(x) for x in v] for k, v in h.history.items()}
        self.n_params_ = int(self.model_.count_params())

    def _predict_raw(self, X):
        return self.model_.predict(X, verbose=0).reshape(-1)

    def _save(self, d: Path):
        self.model_.save(str(d / "model.keras"))
        import json
        (d / "history.json").write_text(json.dumps(getattr(self, "history_", {})))

    def _load(self, d: Path):
        from tensorflow import keras
        self.model_ = keras.models.load_model(str(d / "model.keras"))


class KerasRegressorModel(_Keras):
    task = "regression"


class KerasClassifierModel(_Keras):
    task = "classification"


if __name__ == "__main__":
    from src.training import cli_main
    cli_main("keras")
