import datasets
import pandas as pd

MY_FOLD_CSV = './dataset/BreaKHis_v1/mysplit1.csv'


def dump_myfold_csv():
    """
dump_myfold_csv()函数:
遍历 4 种不同的放大倍率(40, 100, 200, 400)
对于每个放大倍率,生成训练集、验证集和测试集的图像数据
将这些图像数据的相关信息(放大倍率、分组、肿瘤类型和路径)存储到一个列表中
最后将这个列表转换为 pandas DataFrame, 并保存到 MY_FOLD_CSV 指定的 CSV 文件中
    """
    data_list = []

    for magnification in [40, 100, 200, 400]:
        train_dataset_bin = {}
        dev_dataset_bin = {}
        test_dataset_bin = {}
        train_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'train', magnification)
        dev_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'dev', magnification)
        test_dataset_bin[magnification] = datasets.BreaKHis_generate('binary', 'test', magnification)
        for img in train_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp': int(magnification),
                              'grp': 'train',
                              'tumor_class': img.info['tumor_class'],
                              'tumor_type': img.info['tumor_type'],
                              'path': img.path.removeprefix('../data/BreaKHis_v1/')})
        for img in dev_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp': int(magnification),
                              'grp': 'dev',
                              'tumor_class': img.info['tumor_class'],
                              'tumor_type': img.info['tumor_type'],
                              'path': img.path.removeprefix('../data/BreaKHis_v1/')})
        for img in test_dataset_bin[magnification].img_list:
            data_list.append({'mag_grp': int(magnification),
                              'grp': 'test',
                              'tumor_class': img.info['tumor_class'],
                              'tumor_type': img.info['tumor_type'],
                              'path': img.path.removeprefix('../data/BreaKHis_v1/')})

    df = pd.DataFrame(data_list)
    df.to_csv(MY_FOLD_CSV, index=False)


def statistic_myfold():
    """
statistic_myfold()函数:
从 MY_FOLD_CSV 文件中读取之前生成的数据
按放大倍率和肿瘤类型对数据进行分组,统计每个组的图像数量
将统计结果整理成一个透视表,其中行是放大倍率,列是肿瘤类型
为每个放大倍率和肿瘤类型添加总数列和行
最后将这个统计表保存到 tumor_table.csv 文件中
    """
    # 将 CSV 文件加载到 pandas DataFrame 中
    df = pd.read_csv(MY_FOLD_CSV)

    # 按mag_grp和tumor_type对数据进行分组，并计算出现次数
    counts = df.groupby(['mag_grp', 'tumor_type']).size().reset_index(name='count')

    # 透视数据以创建一个表，其中 mag_grp 为行，tumor_type为列
    table = counts.pivot(index='mag_grp', columns='tumor_type', values='count')

    # 用 0 填充任何缺失的值
    table = table.fillna(0)

    # 为每mag_grp中的图像总数添加一列
    table['total'] = table.sum(axis=1)

    # 为每tumor_class中的图像总数添加列
    table['B'] = table['A'] + table['F'] + table['PT'] + table['TA']
    table['M'] = table['DC'] + table['LC'] + table['MC'] + table['PC']

    # 为每tumor_type中的图像总数添加一行
    table.loc['total'] = table.sum(axis=0)

    # 对列重新排序,4种良心肿瘤和4种恶性肿瘤
    table = table[['A(腺病)', 'F(纤维腺瘤)', 'PT(叶状肿瘤)', 'TA(管状腺酮)', 'B()', 'DC()', 'LC()', 'MC()', 'PC()', 'M()', 'total()']]

    # dump the table to a new CSV file
    table.to_csv('tumor_table.csv')


if __name__ == '__main__':
    dump_myfold_csv()
    statistic_myfold()
