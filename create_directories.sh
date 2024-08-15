#!/bin/bash

echo "artifacts file structure creation initiated"

BASE_DIR="artifacts"

mkdir -p $BASE_DIR/Data/raw
mkdir -p $BASE_DIR/Data/clean
mkdir -p $BASE_DIR/Data/preprocessed
mkdir -p $BASE_DIR/Data/prediction

mkdir -p $BASE_DIR/logs/django/admin
mkdir -p $BASE_DIR/logs/django/home_app
mkdir -p $BASE_DIR/logs/django/prediction_app

mkdir -p $BASE_DIR/logs/training/file_operation
mkdir -p $BASE_DIR/logs/training/stage1
mkdir -p $BASE_DIR/logs/training/stage2
mkdir -p $BASE_DIR/logs/training/stage3
mkdir -p $BASE_DIR/logs/training/stage4
mkdir -p $BASE_DIR/logs/training/stage5

mkdir -p $BASE_DIR/logs/prediction/file_operation
mkdir -p $BASE_DIR/logs/prediction/stage1
mkdir -p $BASE_DIR/logs/prediction/stage2
mkdir -p $BASE_DIR/logs/prediction/stage3
mkdir -p $BASE_DIR/logs/prediction/stage4
mkdir -p $BASE_DIR/logs/prediction/stage5

mkdir -p $BASE_DIR/models/prediction_models
mkdir -p $BASE_DIR/models/cluster_models
mkdir -p $BASE_DIR/models/preprocessed_models

mkdir -p $BASE_DIR/plots

echo "Directories created successfully!"
