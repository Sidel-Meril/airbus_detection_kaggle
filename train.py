from LightUNet.model import LightUNet
from LightUNet.helpers import MixedLoss
from  LightUNet.preprocessing import get_train_test
from config import *

if __name__=='__main__':
    train_ds, val_ds = get_train_test(TEST_SIZE)

    model = LightUNet(base_model_name=BASE_MODEL, checkpoints_path=CHECKPOINTS_PATH)
    model.build(input_shape = {'image':(*TARGET_IM_SIZE, 3), 'mask': (*TARGET_IM_SIZE, 1)})

    print(model.summary())

    model.compile(
        optimizer=OPTIMIZER,
        loss = MixedLoss(alpha=ALPHA, gamma=GAMMA),
        learning_rate=LEARNING_RATE
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
    )

    model.save()