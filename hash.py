image1_name = 'image54.jpeg'
image2_name = 'image60.jpeg'
image3_name = 'image56.jpeg'

from PIL import Image
import imagehash

if __name__ == '__main__':
    hash1 = imagehash.dhash(Image.open(image1_name))
    hash2 = imagehash.dhash(Image.open(image2_name))
    print("len(hash1.hash) ** 2", len(hash1.hash), (hash1))
    # 汉明距离
    similarity = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    # dhash 0.984375
    # phash 0.9375
    print(similarity)

