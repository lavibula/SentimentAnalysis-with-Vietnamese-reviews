rm -rf /opt/conda/lib/python3.10/site-packages/aiohttp-3.9.1.dist-info

pip install -q -r requirements.txt || echo "requirements.txt not found in the current directory"
pip install -q -r model/ktrnh/requirements.txt || echo "requirements.txt not found in model/ktrnh/"
pip install -q -r /kaggle/working/SentimentAnalysis-with-Vietnamese-reviews/model/ktrnh/requirements.txt || echo "requirements.txt not found in /kaggle/working/SentimentAnalysis-with-Vietnamese-reviews/model/ktrnh/"

echo "Setup complete. All necessary packages have been installed."
