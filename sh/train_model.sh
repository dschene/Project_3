#!/bin/bash

echo "Enter 'r' to train a restnet model or 'c' to train a custom model: "
read choice

if [ "$choice" = 'r' ]; then
  python ../train_restnet.py
elif [ "$choice" = 'c' ]; then
  python ../train_model.py
else
  echo "Invalid choice"
fi