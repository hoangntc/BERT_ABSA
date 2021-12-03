# !/bin/bash

PROJ_PATH=/home/hoang/github/BERT_ABSA

# echo "Begin: " `date +'%Y-%m-%d %H:%M'`
# echo "Run: RESTAURANT"

# python $PROJ_PATH/src/main.py

echo "Begin: " `date +'%Y-%m-%d %H:%M'`

echo "Run: RESTAURANTS"
python $PROJ_PATH/src/main.py -config_file ../src/config/bert_restaurant_config.json -model_name bert
python $PROJ_PATH/src/main.py -config_file ../src/config/syn_restaurant_config.json -model_name syn
python $PROJ_PATH/src/main.py -config_file ../src/config/sem_restaurant_config.json -model_name synsem

echo "Run: LAPTOPS"
python $PROJ_PATH/src/main.py -config_file ../src/config/bert_laptop_config.json -model_name bert
python $PROJ_PATH/src/main.py -config_file ../src/config/syn_laptop_config.json -model_name syn
python $PROJ_PATH/src/main.py -config_file ../src/config/sem_laptop_config.json -model_name synsem
