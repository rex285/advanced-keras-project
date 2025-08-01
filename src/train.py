import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
import yaml
import os
from datetime import datetime
import numpy as np

def load_config():
    """Load configuration from YAML file"""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_model(config):
    """Build Keras model based on configuration"""
    input_shape = tuple(config['data']['input_shape'])
    num_classes = config['data']['num_classes']
    
    inputs = keras.Input(shape=input_shape)
    
    # Data augmentation
    if config['augmentation']['enabled']:
        x = layers.RandomRotation(config['augmentation']['rotation_range'])(inputs)
        x = layers.RandomZoom(config['augmentation']['zoom_range'])(x)
        x = layers.RandomFlip(config['augmentation']['flip_mode'])(x)
    else:
        x = inputs
    
    # Model architecture
    for i, filters in enumerate(config['model']['conv_filters']):
        x = layers.Conv2D(filters, config['model']['kernel_size'], activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        if i < len(config['model']['conv_filters']) - 1:  # No pooling after last conv layer
            x = layers.MaxPooling2D(config['model']['pool_size'])(x)
            x = layers.Dropout(config['model']['dropout_rate'])(x)
    
    x = layers.Flatten()(x)
    for units in config['model']['dense_units']:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config['model']['dense_dropout_rate'])(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = optimizers.Adam(learning_rate=config['training']['initial_learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(config):
    """Create training callbacks"""
    callbacks_list = []
    
    # Model checkpoint
    os.makedirs('models', exist_ok=True)
    checkpoint_path = os.path.join('models', 'best_model.h5')
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config['training']['monitor'],
        save_best_only=True,
        save_weights_only=False,
        mode='auto'
    )
    callbacks_list.append(model_checkpoint)
    
    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor=config['training']['monitor'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        restore_best_weights=True
    )
    callbacks_list.append(early_stopping)
    
    # TensorBoard
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list.append(tensorboard)
    
    return callbacks_list

def main():
    """Main training function"""
    config = load_config()
    
    # Load data (placeholder - replace with your data loading)
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    
    # Build and train model
    model = build_model(config)
    model.summary()
    
    callbacks_list = get_callbacks(config)
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join('models', 'final_model.h5'))

if __name__ == '__main__':
    main()
