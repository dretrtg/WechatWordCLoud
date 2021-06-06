import pandas as pd
import jieba
import codecs
from imageio import imread
import matplotlib.pyplot as plt
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud


def load_file_and_seg(file_path):
    """导入文件并分词"""
    f = codecs.open(file_path, encoding='utf-8')
    content = f.read()
    f.close()
    segment = []
    segs = jieba.cut(content)
    for seg in segs:
        if len(seg) > 1 and seg != '\r\n':
            segment.append(seg)
    return segment


def get_words_count(file_path, stopwords_file=None):
    """统计词频"""
    segment = load_file_and_seg(file_path)
    df = pd.DataFrame({'segment': segment})
    if stopwords_file:
        stopwords = pd.read_csv(stopwords_file, index_col=False, quoting=3, sep="\t",
                                names=['stopword'], encoding="utf-8")
        df = df[~df.segment.isin(stopwords.stopword)]
    words_count = df['segment'].value_counts()
    return words_count.to_dict()


def has_chinese(string):
    """字符串中是否含有中文"""
    for char in string:
        if u'\u4e00' <= char <= u'\u9fa5':
            return True
    return False


def preprocessing(file_path):
    """数据的预处理"""
    df = pd.read_csv(file_path, sep=",")
    keep_indexes = []
    for i in range(df.shape[0]):
        txt = df.loc[i, 'content']
        if txt.startswith('<msg>'):
            continue
        if txt.startswith("欢迎你再次回到微信"):
            continue
        if has_chinese(txt):
            keep_indexes.append(i)

    df2 = df.iloc[keep_indexes, :]
    df2.sort_values(by='createTime', inplace=True)
    df2 = df2.content

    file = open('1.txt', 'w+', encoding='utf-8')
    for i in df2:
        file.write(i + '\n')
    file.close()


if __name__ == '__main__':
    file = '1.csv'
    preprocessing(file)
    words_count = get_words_count('./1.txt')
    bimg = imread('./1.jpg')  # 使用一张图片为模板
    wordcloud = WordCloud(background_color='white', mask=bimg, font_path='simsun.ttc')  # 注意字体
    wordcloud = wordcloud.fit_words(words_count)
    bimgColors = ImageColorGenerator(bimg)
    plt.axis("off")
    plt.imshow(wordcloud.recolor(color_func=bimgColors))
    plt.show()