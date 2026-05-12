# HorusLM


#### To train the model, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed/aes_clean/ramses_clean/mixed_clean]
```

#### To test the trained model on test set, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed/aes_clean/ramses_clean/mixed_clean] --mode decode
```

#### To interact with the trained model in real time, run the following command:

```
python main.py --cuda True --dataset [aes/ramses/mixed/aes_clean/ramses_clean/mixed_clean] --mode realtime
```
