python run.py train-val --epochs 10 --seed 123 --use-class-weights --weight-decay 0.01 --lr 5e-5 --name base 
python run.py train-val --epochs 10 --seed 123 --weight-decay 0.01 --lr 5e-5 --name no-weighting


False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
2499/2499 [==============================] - 2451s 979ms/step - loss: 0.7931 - accuracy: 0.7261 - val_loss: 0.6953 - val_accuracy: 0.7406
Epoch 2/10
2499/2499 [==============================] - 2446s 979ms/step - loss: 0.6936 - accuracy: 0.7363 - val_loss: 0.6465 - val_accuracy: 0.7589
Epoch 3/10
2499/2499 [==============================] - 2441s 977ms/step - loss: 0.6523 - accuracy: 0.7509 - val_loss: 0.6187 - val_accuracy: 0.7752
Epoch 4/10
2499/2499 [==============================] - 2440s 976ms/step - loss: 0.6228 - accuracy: 0.7604 - val_loss: 0.5878 - val_accuracy: 0.7812
Epoch 5/10
2499/2499 [==============================] - 2441s 977ms/step - loss: 0.6065 - accuracy: 0.7665 - val_loss: 0.5611 - val_accuracy: 0.7938
Epoch 6/10
2499/2499 [==============================] - 2476s 991ms/step - loss: 0.5888 - accuracy: 0.7750 - val_loss: 0.5437 - val_accuracy: 0.8010
Epoch 7/10
2499/2499 [==============================] - 2548s 1s/step - loss: 0.5742 - accuracy: 0.7792 - val_loss: 0.5383 - val_accuracy: 0.8045
Epoch 8/10
2499/2499 [==============================] - 2543s 1s/step - loss: 0.5601 - accuracy: 0.7873 - val_loss: 0.5443 - val_accuracy: 0.8024
Epoch 9/10
2499/2499 [==============================] - 2544s 1s/step - loss: 0.5553 - accuracy: 0.7871 - val_loss: 0.5288 - val_accuracy: 0.8067
Epoch 10/10
2499/2499 [==============================] - 2542s 1s/step - loss: 0.5429 - accuracy: 0.7938 - val_loss: 0.5229 - val_accuracy: 0.8142
