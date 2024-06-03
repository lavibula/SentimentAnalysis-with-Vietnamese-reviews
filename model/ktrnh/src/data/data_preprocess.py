import re
import unicodedata
from pyvi import ViTokenizer

class TextPreprocessor:
    def __init__(self):
        self.acronyms = {
            'ô kêi':'ok','okie':'ok','o kê':'ok',
            'okey':'ok','ôkê':'ok','oki':'ok','oke': 'ok','okay':'ok','okê':'ok',
            'tks': 'cám ơn','thks': 'cám ơn','thanks': 'cám ơn','ths': 'cám ơn','thank': 'cám ơn',
            '⭐':'star','*':'star','🌟':'star',
            'kg': 'không','not': 'không', 'kg': 'không','"k': 'không','kh':'không','kô':'không','hok':'không','kp': 'không phải','kô': 'không','"ko': 'không', 'ko': 'không', 'k': 'không','khong': 'không', 'hok': 'không',
            'cute': 'dễ thương','vs': 'với','wa':'quá','wá': 'quá','j': 'gì','“':'',
            'sz': 'cỡ','size': 'cỡ', 'đx': 'được','dk': 'được','dc': 'được','đk': 'được',
            'đc': 'được','authentic': 'chuẩn chính hãng','aut': 'chuẩn chính hãng', 'auth': 'chuẩn chính hãng','store': 'cửa hàng',
            'shop': 'cửa hàng','sp': 'sản phẩm','gud': 'tốt','god': 'tốt','wel done':'tốt','good': 'tốt','gút': 'tốt',
            'sấu': 'xấu','gut': 'tốt', 'tot': 'tốt', 'nice': 'tốt','perfect':'rất tốt','bt': 'bình thường',
            'time': 'thời gian','qá': 'quá', 'ship': 'giao hàng', 'm': 'mình', 'mik': 'mình',
            'ể':'ể','product':'sản phẩm','quality':'chất lượng','chat':'chất','excelent':'hoàn hảo','bad':'tệ','fresh':'tươi','sad':'tệ',
            'date': 'hạn sử dụng','hsd': 'hạn sử dụng','quickly': 'nhanh','quick': 'nhanh','fast': 'nhanh','delivery': 'giao hàng','síp': 'giao hàng',
            'beautiful': 'đẹp tuyệt vời', 'tl': 'trả lời', 'r': 'rồi', 'shopE': 'cửa hàng','order': 'đặt hàng',
            'chất lg': 'chất lượng','sd': 'sử dụng','dt': 'điện thoại','nt': 'nhắn tin','tl': 'trả lời','sài': 'xài','bjo':'bao giờ',
            'thick': 'thích','thik': 'thích', 'sop': 'cửa hàng', 'shop': 'cửa hàng', 
            'fb':'facebook','face':'facebook','very': 'rất','quả ng':'quảng ',
            'dep': 'đẹp','xau': 'xấu','delicious': 'ngon', 'hàg': 'hàng', 'qủa': 'quả',
            'iu': 'yêu','fake': 'giả mạo','trl':'trả lời',
            'por': 'tệ','poor': 'tệ','ib':'nhắn tin','rep':'trả lời','fback':'feedback','fedback':'feedback',
            'max': 'cực kỳ',
            'full':'đầy đủ', 'ful':'đầy đủ'
        }
        self.not_words = {"không", "không hề", "chẳng", "chưa", "không phải", "chả", "mất",
                          "thiếu", "đếch", "đéo", "kém", "nỏ", "not",
                          "bớt", "không bao giờ", "chưa bao giờ"}
        self.not_words = sorted(self.not_words, key=len, reverse=True)
        self.replacements = {
            'a':'àáảãạăằắẳẵặâầấẩẫậ',
            'e':'èéẻẽẹêềếểễệ',
            'i':'ìíỉĩị',
            'o':'òóỏõọôồốổỗộơờớởỡợ',
            'u':'ùúủũụưừứửữự',
            'y':'ỳýỷỹỵ',
            'd':'đ',
            'A':'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ',
            'E':'ÈÉẺẼẸÊỀẾỂỄỆ',
            'I':'ÌÍỈĨỊ',
            'O':'ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ',
            'U':'ÙÚỦŨỤƯỪỨỬỮỰ',
            'Y':'ỲÝỶỸỴ',
            'D':'Đ'
        }

    def remove_HTML(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def convert_unicode(self, text):
        return unicodedata.normalize('NFC', text)

    def remove_elongated_chars(self, text):
        for char, replacements_str in self.replacements.items():
            pattern = rf"({char})\1+"
            text = re.sub(pattern, char, text)
        
        pattern = r"(\w)\1+"
        text = re.sub(pattern, r'\1', text)
        return text

    def handle_negation(self, text):
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in self.not_words) + r')\b'
        text = re.sub(pattern, 'không', text, flags=re.IGNORECASE)
        return text

    def normalize_acronyms(self, text):
        words = text.split()
        normalized_text = ' '.join([self.acronyms.get(word.lower(), word) for word in words])
        return normalized_text

    def word_segmentation(self, text):
        return ViTokenizer.tokenize(text)

    def remove_unnecessary_characters(self, text):
        text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
        text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ các ký tự đặc biệt
        return text.strip()

    def preprocess(self, text):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")
        text = self.remove_HTML(text)
        text = self.normalize_acronyms(text)
        text = self.convert_unicode(text)
        text = self.remove_elongated_chars(text)
        text = self.handle_negation(text)
        text = self.word_segmentation(text)
        text = self.remove_unnecessary_characters(text)
        return text
