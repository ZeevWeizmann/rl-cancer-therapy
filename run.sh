echo "=============================="
echo "TRAINING RL MODELS"
echo "=============================="

python training/train_dqn.py
python training/train_ppo.py

echo ""
echo "=============================="
echo "EVALUATING POLICIES"
echo "=============================="

python evaluation/compare_models.py

echo ""
echo "=============================="
echo "DONE"
echo "=============================="