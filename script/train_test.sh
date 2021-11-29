# !/bin/bash

PROJ_PATH=/home/hoang/github/BERT_ABSA

# echo "Begin: " `date +'%Y-%m-%d %H:%M'`
# echo "Run: RESTAURANT"

# python $PROJ_PATH/src/main.py

echo "Begin: " `date +'%Y-%m-%d %H:%M'`

echo "Run: RESTAURANTS"
python $PROJ_PATH/src/main.py -config_file ../src/config/restaurant_config.json -model_name bert

echo "Run: LAPTOPS"
python $PROJ_PATH/src/main.py -config_file ../src/config/laptop_config.json -model_name bert