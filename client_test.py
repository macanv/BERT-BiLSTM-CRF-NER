# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/29 14:32
 @Author  : MaCan (ma_cancan@163.com)
 @File    : client_test.py
"""
import time
from bert_base.client import BertClient


def ner_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        # rst = bc.encode([list(str1)], is_tokenized=True)
        # str1 = list(str1)
        rst = bc.encode([str1], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        print(time.perf_counter() - start_t)


def ner_cu_seg():
    """
    自定义分字
    :return:
    """
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
        start_t = time.perf_counter()
        str1 = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
        rst = bc.encode([list(str1)], is_tokenized=True)
        print('rst:', rst)
        print(len(rst[0]))
        print(time.perf_counter() - start_t)


def class_test():
    with BertClient(show_server_config=False, check_version=False, check_length=False, mode='CLASS') as bc:
        start_t = time.perf_counter()
        str = '北京时间2月17日凌晨，第69届柏林国际电影节公布主竞赛单元获奖名单，王景春、咏梅凭借王小帅执导的中国影片《地久天长》连夺最佳男女演员双银熊大奖，这是中国演员首次包揽柏林电影节最佳男女演员奖，为华语影片刷新纪录。与此同时，由青年导演王丽娜执导的影片《第一次的别离》也荣获了本届柏林电影节新生代单元国际评审团最佳影片，可以说，在经历数个获奖小年之后，中国电影在柏林影展再次迎来了高光时刻。'
        str2 = '受粤港澳大湾区规划纲要提振，港股周二高开，恒指开盘上涨近百点，涨幅0.33%，报28440.49点，相关概念股亦集体上涨，电子元件、新能源车、保险、基建概念多数上涨。粤泰股份、珠江实业、深天地A等10余股涨停；中兴通讯、丘钛科技、舜宇光学分别高开1.4%、4.3%、1.6%。比亚迪电子、比亚迪股份、光宇国际分别高开1.7%、1.2%、1%。越秀交通基建涨近2%，粤海投资、碧桂园等多股涨超1%。其他方面，日本软银集团股价上涨超0.4%，推动日经225和东证指数齐齐高开，但随后均回吐涨幅转跌东证指数跌0.2%，日经225指数跌0.11%，报21258.4点。受芯片制造商SK海力士股价下跌1.34％拖累，韩国综指下跌0.34％至2203.9点。澳大利亚ASX 200指数早盘上涨0.39％至6089.8点，大多数行业板块均现涨势。在保健品品牌澳佳宝下调下半财年的销售预期后，其股价暴跌超过23％。澳佳宝CEO亨弗里（Richard Henfrey）认为，公司下半年的利润可能会低于上半年，主要是受到销售额疲弱的影响。同时，亚市早盘澳洲联储公布了2月会议纪要，政策委员将继续谨慎评估经济增长前景，因前景充满不确定性的影响，稳定当前的利率水平比贸然调整利率更为合适，而且当前利率水平将有利于趋向通胀目标及改善就业，当前劳动力市场数据表现强势于其他经济数据。另一方面，经济增长前景亦令消费者消费意愿下滑，如果房价出现下滑，消费可能会进一步疲弱。在澳洲联储公布会议纪要后，澳元兑美元下跌近30点，报0.7120 。美元指数在昨日触及96.65附近的低点之后反弹至96.904。日元兑美元报110.56，接近上一交易日的低点。'
        str3 = '新京报快讯 据国家市场监管总局消息，针对媒体报道水饺等猪肉制品检出非洲猪瘟病毒核酸阳性问题，市场监管总局、农业农村部已要求企业立即追溯猪肉原料来源并对猪肉制品进行了处置。两部门已派出联合督查组调查核实相关情况，要求猪肉制品生产企业进一步加强对猪肉原料的管控，落实检验检疫票证查验规定，完善非洲猪瘟检测和复核制度，防止染疫猪肉原料进入食品加工环节。市场监管总局、农业农村部等部门要求各地全面落实防控责任，强化防控措施，规范信息报告和发布，对不按要求履行防控责任的企业，一旦发现将严厉查处。专家认为，非洲猪瘟不是人畜共患病，虽然对猪有致命危险，但对人没有任何危害，属于只传猪不传人型病毒，不会影响食品安全。开展猪肉制品病毒核酸检测，可为防控溯源工作提供线索。'
        rst = bc.encode([str, str2, str3])
        print('rst:', rst)
        print('time used:{}'.format(time.perf_counter() - start_t))


if __name__ == '__main__':
    # class_test()
    ner_test()
    ner_cu_seg()