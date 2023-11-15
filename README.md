# cs415_final
## final project for uic computer vision

### To clean the images

1. Go to the kaggle link and download the data -->https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities?select=Montgomery
2. Move it to a place you want on your computer
3. Delete the Montgomery data, and only keep the `CXR_png` data
4. Change the `image_path` variable in the `standardize_data` script to match where you stored the data
5. Run the script. The new, cleaned images should be in a folder called `clean_images`. If you see this histogram it worked
   ![image](https://github.com/monkeygobah/cs415_final/assets/117255104/d0cbc143-aa51-48ee-a189-d24931daf6f9)
