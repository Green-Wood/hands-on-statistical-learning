from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score


def news_predict(train_sample, train_label, test_sample):
    """
    训练模型并进行预测，返回预测结果
    :param train_sample:原始训练集中的新闻文本，类型为ndarray
    :param train_label:训练集中新闻文本对应的主题标签，类型为ndarray
    :param test_sample:原始测试集中的新闻文本，类型为ndarray
    :return 预测结果，类型为ndarray
    """

    Tfid = TfidfVectorizer()
    train_x = Tfid.fit_transform(train_sample)
    test_x = Tfid.transform(test_sample)

    clf = MultinomialNB(alpha=3)
    clf.fit(train_x, train_label)

    test_pred = clf.predict(test_x)
    return test_pred


if __name__ == '__main__':
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    train_sample = train_data.data
    train_label = train_data.target
    test_sample = test_data.data
    test_label = test_data.target

    test_pred = news_predict(train_sample, train_label, test_sample)
    print(accuracy_score(test_label, test_pred))