#!/bin/bash
source ~/venv/bin/activate
( nohup python main.py 20 exp2_feed_6 feed_foward_3 sigmoid exp2 ; 
nohup python main.py 40 exp2_feed_6 feed_foward_3 sigmoid exp ;
nohup python main.py 60 exp2_feed_6 feed_foward_3 sigmoid exp ; ) &