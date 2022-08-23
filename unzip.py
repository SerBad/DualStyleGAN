import os
import subprocess
import time
import zipfile
import shutil
import argparse


class BuildOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument("--zip", type=str, required=True, help="zip路径")
        self.parser.add_argument("--path", type=str, required=True, help="保存文件的路径")
        self.parser.add_argument("--n", default=False, action='store_true', required=False, help="覆盖文件")

    def parse(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    parser = BuildOptions()
    args = parser.parse()

    print('zip', args.zip)
    dir_name = os.path.dirname(args.path)
    zFile = zipfile.ZipFile(args.zip, "r")

    for fileM in zFile.namelist():
        try:
            new_zip_file = fileM.encode('cp437').decode('utf-8')
        except Exception as e:
            print(e)
            new_zip_file = fileM.encode('cp437').decode('gbk')

        file = os.path.join(args.path, new_zip_file)

        if args.n:
            if not os.path.exists(file):
                print("extract", file)
                zFile.extract(fileM, args.path)
                os.rename(os.path.join(args.path, fileM), file)
        else:
            zFile.extract(fileM, args.path)
            os.rename(os.path.join(args.path, fileM), file)

    zFile.close()
