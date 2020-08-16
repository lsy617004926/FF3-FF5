import pandas as pd
import numpy as np

daily = pd.read_csv('./daily.csv', encoding='cp949', index_col=0, parse_dates=True)
monthly = pd.read_csv('./monthly.csv', encoding='cp949', index_col=0, parse_dates=True)
annual = pd.read_csv('./annual.csv', encoding='cp949', index_col=0, parse_dates=True)
#quarterly = pd.read_csv('./quarterly.csv', encoding='cp949', index_col=0, parse_dates=True)
#quarterly = quarterly.iloc[:,:-1]
#book = pd.read_csv('./book.csv', encoding='cp949', index_col=0).iloc[:-1]


def get_data(data, name, ind=None, mask=None):
    result = data[data.iloc[:, 0] == name].iloc[:, 1:].T.dropna(how='all')
    result.index = pd.to_datetime(result.index)

    if not isinstance(ind, type(None)):
        result.index = ind

    if not isinstance(mask, type(None)):
        result = result[mask]

    return result

adj_price = get_data(monthly, '수정주가').astype('float')
mkt_equity = get_data(monthly, '시가총액(전체)').astype('float')
mkt_where = get_data(monthly, '상장된 시장')
industry = get_data(monthly, 'WICS업종명(중)').replace(['은행','보험','증권','다각화된금융','부동산'],np.nan)
industry = industry.replace(industry.iloc[-1].dropna().unique(), True)
industry = industry.replace(np.nan, False).astype('bool')

june = [x for x in adj_price.index if x.month == 6]
quarter = [x for x in adj_price.loc['2000':].index if x.month == 3 or x.month == 6 or x.month == 9 or x.month == 12][1:]
book.index = quarter
book_mask = book.applymap(lambda x: True if x>0 else False).astype('bool')
industry = industry & book_mask.reindex(industry.index).fillna(method='ffill', limit=12)
kospi_who = mkt_where.applymap(lambda x: True if x == 'KOSPI' else False).astype('bool')



sec = get_data(quarterly, '보통주자본금', ind=quarter, mask=industry)
#pstk = get_data(annual, '우선주자본금', mask=industry).replace(np.nan, 0)
txditc = get_data(quarterly, '이연법인세자산', ind=quarter, mask=industry).replace(np.nan, 0)
oper_inc = get_data(quarterly, '영업이익', ind=quarter, mask=industry)
tot_asset = get_data(quarterly, '자산총계', ind=quarter, mask=industry)
#xrds = pd.read_csv('./xrd.csv', encoding='cp949', index_col=0, parse_dates=True)
#xrd = get_data(xrds, '연구개발비', ind=june, mask=industry).astype('float')

#xrd_me = xrd.divide(mkt_equity.shift(6).replace(0, np.nan), axis=0).dropna(how='all')

book_equity = pd.read_csv('./book_y.csv', index_col=0).iloc[:-1]
book_equity.index = june
book_equity = book_equity[industry]


daily_price = daily[daily.iloc[:,0] == '수정주가'].iloc[:,1:].T
daily_price.index = pd.to_datetime(daily_price.index)
daily_rtn = daily_price.diff(1).divide(daily_price.shift(1),axis=0)

monthly_rtn = adj_price.diff(1).divide(adj_price.shift(1).replace(0, np.nan), axis=0).dropna(how='all').replace(0, np.nan)
market_cap = mkt_equity.loc[june]
market_cap = market_cap[industry]
bm = book_equity.divide(market_cap.shift(1).replace(0, np.nan), axis=0).dropna(how='all').astype('float')
bm = bm[bm>0]


op = oper_inc.divide(book_equity.replace(0, np.nan), axis=0)
op = op[bm > 0]
inv = tot_asset.diff(12).divide(tot_asset.shift(12).replace(0, np.nan), axis=0).dropna(how='all')
inv = inv[bm > 0]

mom = monthly_rtn.rolling(12, min_periods=12).apply(lambda x: (1+x.iloc[:-1]).prod()-1)



def sort_2(data, kospi_who, label):
    result = data.copy().astype('str')
    result[:] = ''
    medi = data[kospi_who].apply(lambda x: x.median(), axis=1)
    result[data.gt(medi,axis=0)] = label[1]
    result[data.le(medi,axis=0)] = label[0]

    return result.replace('', np.nan)

size_sort = sort_2(market_cap, kospi_who, ['S', 'B'])

market_cap_m = mkt_equity[industry]
book_m = book_equity.reindex(market_cap_m.index).fillna(method='ffill', limit=12)
bm_m = book_m.divide(market_cap_m.replace(0, np.nan), axis=0).dropna(how='all').astype('float')
bm_m = bm_m[bm_m>0]


size_sort_m = sort_2(market_cap_m, kospi_who, ['S', 'B'])



def divide_percentile(size, fac, kospi, lb, ub, name_list):
    result = size.astype('object').copy()
    crit_lb = fac[kospi].apply(lambda x: np.nanpercentile(x, lb), axis=1)
    crit_ub = fac[kospi].apply(lambda x: np.nanpercentile(x, ub), axis=1)

    lb = fac.lt(crit_lb,axis=0)
    neu = fac.ge(crit_lb, axis=0) & fac.lt(crit_ub, axis=0)
    ub = fac.ge(crit_ub, axis=0)

    result[lb] = result[lb].applymap(lambda x: x+name_list[0] if isinstance(x, str) else x)
    result[neu] = result[neu].applymap(lambda x: x + name_list[1] if isinstance(x, str) else x)
    result[ub] = result[ub].applymap(lambda x: x + name_list[2] if isinstance(x, str) else x)

    result = result.applymap(lambda x: np.nan if isinstance(x, str) and len(x) == 1 else x)

    return result


def get_rtn(x, result, monthly_rtn, market_cap):
    i = market_cap.index[1].month - market_cap.index[0].month

    if i == 0:
        i = 12

    i -= 1

    year = x.name.year
    month = x.name.month

    start_m = (month + 1) % 12
    end_m = (start_m + i) % 12

    if start_m == 0:
        start_m = 12

    if end_m == 0:
        end_m = 12

    start_y = year + (month+1)//13
    end_y = year + (month+i+1)//13

    start_d = str(start_y) + '-' + str(start_m)
    end_d = str(end_y) + '-' + str(end_m)
    unique_v = x.unique()
    for g in unique_v:
        tic_list = x[x == g].index
        tmp_mv = market_cap.loc[x.name.strftime('%Y-%m'), tic_list].iloc[0]
        tmp_rtn = monthly_rtn.loc[start_d:end_d, tic_list]
        tmp_mv /= tmp_mv.sum()
        result.loc[start_d:end_d,g] = (tmp_rtn*tmp_mv).sum(skipna=True, axis=1)

    return 'Done!'



def fac_port_rtn(group, monthly_rtn, market_cap):
    stock_list = group.dropna(how='all')
    port_rtn = pd.DataFrame(index=monthly_rtn.index, columns=stock_list.iloc[0].dropna().unique())
    tmp_df = stock_list.apply(lambda x: get_rtn(x.dropna(), port_rtn, monthly_rtn, market_cap), axis=1)

    return port_rtn.dropna(axis=0)


# ff3 & ff5 2x3

#for only kospi
#size_sort = size_sort[kospi_who]

size_bm = divide_percentile(size_sort, bm.astype('float'), kospi_who, 30, 70, ['G', 'N', 'V'])

size_op = divide_percentile(size_sort, op.astype('float'), kospi_who, 30, 70, ['W', 'N', 'R'])
size_inv = divide_percentile(size_sort, inv.astype('float'), kospi_who, 30, 70, ['C', 'N', 'A'])

size_mom = divide_percentile(size_sort_m, mom.astype('float'), kospi_who, 30, 70, ['D', 'N', 'U'])

size_hml_devil = divide_percentile(size_sort_m, bm_m, kospi_who, 30, 70, ['G', 'N', 'V'])
#size_xrd = divide_percentile(size_sort, xrd_me, kospi_who, 30, 70, ['L', 'N', 'H'])



size_bm_port = fac_port_rtn(size_bm, monthly_rtn, market_cap)
size_bm_port_d = fac_port_rtn(size_bm, daily_rtn, market_cap)

size_op_port = fac_port_rtn(size_op, monthly_rtn, market_cap)
size_inv_port = fac_port_rtn(size_inv, monthly_rtn, market_cap)

size_mom_port = fac_port_rtn(size_mom, monthly_rtn, market_cap_m)
size_mom_port_d = fac_port_rtn(size_mom, daily_rtn, market_cap_m)

size_devil_port = fac_port_rtn(size_hml_devil, monthly_rtn, market_cap_m)
size_devil_port_d = fac_port_rtn(size_hml_devil, daily_rtn, market_cap_m)

#size_xrd_port = fac_port_rtn(size_xrd, monthly_rtn, market_cap)



smb_bm = (size_bm_port['SV']+size_bm_port['SN']+size_bm_port['SG'])/3 - (size_bm_port['BV']+size_bm_port['BN']+size_bm_port['BG'])/3
smb_bm_d = (size_bm_port_d['SV']+size_bm_port_d['SN']+size_bm_port_d['SG'])/3 - (size_bm_port_d['BV']+size_bm_port_d['BN']+size_bm_port_d['BG'])/3

smb_op = (size_op_port['SR']+size_op_port['SN']+size_op_port['SW'])/3 - (size_op_port['BR']+size_op_port['BN']+size_op_port['BW'])/3
smb_inv = (size_inv_port['SC']+size_inv_port['SN']+size_inv_port['SA'])/3 - (size_inv_port['BC']+size_inv_port['BN']+size_inv_port['BA'])/3

smb_devil = (size_devil_port['SV']+size_devil_port['SN']+size_devil_port['SG'])/3 - (size_devil_port['BV']+size_devil_port['BN']+size_devil_port['BG'])/3
smb_devil_d = (size_devil_port_d['SV']+size_devil_port_d['SN']+size_devil_port_d['SG'])/3 - (size_devil_port_d['BV']+size_devil_port_d['BN']+size_devil_port_d['BG'])/3

smb_ff5 = (smb_bm+smb_op+smb_inv)/3

hml = (size_bm_port['SV']+size_bm_port['BV'])/2 - (size_bm_port['SG']+size_bm_port['BG'])/2
hml_d = (size_bm_port_d['SV']+size_bm_port_d['BV'])/2 - (size_bm_port_d['SG']+size_bm_port_d['BG'])/2
hml_devil = (size_devil_port['SV']+size_devil_port['BV'])/2 - (size_devil_port['SG']+size_devil_port['BG'])/2
hml_devil_d = (size_devil_port_d['SV']+size_devil_port_d['BV'])/2 - (size_devil_port_d['SG']+size_devil_port_d['BG'])/2


hml_s = size_bm_port['SV'] - size_bm_port['SG']
hml_b = size_bm_port['BV'] - size_bm_port['BG']

rmw = (size_op_port['SR']+size_op_port['BR'])/2 - (size_op_port['SW']+size_op_port['BW'])/2
rmw_s = size_op_port['SR'] - size_op_port['SW']
rmw_b = size_op_port['BR'] - size_op_port['BW']

cma = (size_inv_port['SC']+size_inv_port['BC'])/2 - (size_inv_port['SA']+size_inv_port['BA'])/2
cma_s = size_inv_port['SC'] - size_inv_port['SA']
cma_b = size_inv_port['BC'] - size_inv_port['BA']

mom_fac = (size_mom_port['BU']+size_mom_port['SU'])/2 - (size_mom_port['BD']+size_mom_port['SD'])/2
mom_fac_d = (size_mom_port_d['BU']+size_mom_port_d['SU'])/2 - (size_mom_port_d['BD']+size_mom_port_d['SD'])/2


#hx_lx = (size_xrd_port['SH']+size_xrd_port['BH'])/2 - (size_xrd_port['SL']+size_xrd_port['BL'])/2
#hx_lx.name = 'HXMLX'


smb_bm.name = 'SMB'
smb_ff5.name = 'SMB'

hml.name = 'HML'
rmw.name = 'RMW'
cma.name = 'CMA'
mom_fac.name = 'MOM'

smb_devil.name = 'SMB_devil'
hml_devil.name = 'HML_devil'

smb_devil_d.name = 'SMB_devil'
hml_devil_d.name = 'HML_devil'

mkt_rf = pd.read_csv('./mkt_rf.csv', encoding='cp949', index_col=0).astype('float')
mkt_rf.columns = adj_price.index

kospi = mkt_rf.loc['코스피'].T
mkt_rtn = kospi.diff(1).divide(kospi.shift(1).replace(0, np.nan), axis=0).dropna(how='all')
rf = mkt_rf.loc['시장금리:통화안정(364일)(%)'].T
rf /= 100
rf = (1+rf)**(1/12)-1

rf.name = 'Rf'

mkt_fac = mkt_rtn-rf
mkt_fac.name = 'Mkt-Rf'
mkt_fac = mkt_fac.dropna()


mkt_d = pd.read_csv('./mkt_rf_d.csv', encoding='cp949', index_col=0, parse_dates=True).astype('float')
mkt_d['MKT-Rf'] = mkt_d['MKT'] - mkt_d['Rf']



ff4_monthly = pd.concat([mkt_fac,smb_bm,hml,mom_fac,rf],axis=1).dropna()
ff4_monthly.to_csv('./ff4_monthly.csv',encoding='cp949')


smb_bm_d.name = 'SMB'
hml_d.name = 'HML'
mom_fac_d.name = 'MOM'

ff4_daily = pd.concat([mkt_d['MKT-Rf'],smb_bm_d, hml_d, mom_fac_d, mkt_d['Rf']],axis=1).replace(0,np.nan).dropna()
ff4_daily.to_csv('./ff4_daily.csv',encoding='cp949')

ff4_devil_monthly = pd.concat([mkt_fac,smb_devil,hml_devil,mom_fac,rf],axis=1).dropna()
ff4_devil_monthly.to_csv('./ff4_devil_monthly.csv', encoding='cp949')

ff4_devil_daily = pd.concat([mkt_d['MKT-Rf'],smb_devil_d, hml_devil_d, mom_fac_d, mkt_d['Rf']],axis=1).replace(0,np.nan).dropna()
ff4_devil_daily.to_csv('./ff4_devil_daily.csv',encoding='cp949')


ff5_monthly = pd.concat([mkt_fac,smb_ff5,hml,rmw,cma,rf],axis=1).dropna()
ff5_monthly.to_csv('./ff5_monthly_q.csv',encoding='cp949')

#ff5_monthly2 = pd.concat([mkt_fac,smb_ff5,hml,rmw,cma,hx_lx,rf],axis=1).dropna()

#ff5_monthly2.loc['2000':,'HXMLX'].astype('float')


#2x2

bm_sort = sort_2(bm, kospi_who, ['L', 'H'])
op_sort = sort_2(op, kospi_who, ['W', 'R'])
inv_sort = sort_2(inv, kospi_who, ['C', 'A'])

size_bm_2x2 = size_sort + bm_sort
size_op_2x2 = size_sort + op_sort
size_inv_2x2 = size_sort + inv_sort

size_bm_2x2_port = fac_port_rtn(size_bm_2x2, monthly_rtn, market_cap)
size_op_2x2_port = fac_port_rtn(size_op_2x2, monthly_rtn, market_cap)
size_inv_2x2_port = fac_port_rtn(size_inv_2x2, monthly_rtn, market_cap)

size_2x2_ports = pd.concat([size_bm_2x2_port, size_op_2x2_port, size_inv_2x2_port],axis=1)


def make_factor(port_rtn, label, ind):
    result = port_rtn[[x for x in port_rtn if x[ind] == label[0]]].mean(axis=1) \
             - port_rtn[[x for x in port_rtn if x[ind] == label[1]]].mean(axis=1)
    return result


smb_2x2 = make_factor(size_2x2_ports, ['S', 'B'], 0)

hml_2x2 = make_factor(size_bm_2x2_port, ['H', 'L'], 1)
hml_s_2x2 = make_factor(size_bm_2x2_port[[x for x in size_bm_2x2_port.columns if x[0] == 'S']], ['H', 'L'], 1)
hml_b_2x2 = make_factor(size_bm_2x2_port[[x for x in size_bm_2x2_port.columns if x[0] == 'B']], ['H', 'L'], 1)

rmw_2x2 = make_factor(size_op_2x2_port, ['R', 'W'], 1)
rmw_s_2x2 = make_factor(size_op_2x2_port[[x for x in size_op_2x2_port.columns if x[0] == 'S']], ['R', 'W'], 1)
rmw_b_2x2 = make_factor(size_op_2x2_port[[x for x in size_op_2x2_port.columns if x[0] == 'B']], ['R', 'W'], 1)

cma_2x2 = make_factor(size_inv_2x2_port, ['C', 'A'], 1)
cma_s_2x2 = make_factor(size_inv_2x2_port[[x for x in size_inv_2x2_port.columns if x[0] == 'S']], ['C', 'A'], 1)
cma_b_2x2 = make_factor(size_inv_2x2_port[[x for x in size_inv_2x2_port.columns if x[0] == 'B']], ['C', 'A'], 1)


#2x2x2x2

sort_2x2x2x2 = size_sort + bm_sort + op_sort + inv_sort
sort_2x2x2x2_port = fac_port_rtn(sort_2x2x2x2, monthly_rtn, market_cap)
smb_2x2x2x2 = make_factor(sort_2x2x2x2_port, ['S', 'B'], 0)

hml_2x2x2x2 = make_factor(sort_2x2x2x2_port, ['H', 'L'], 1)
hml_s_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'S']], ['H', 'L'], 1)
hml_b_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'B']], ['H', 'L'], 1)

rmw_2x2x2x2 = make_factor(sort_2x2x2x2_port, ['R', 'W'], 2)
rmw_s_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'S']], ['R', 'W'], 2)
rmw_b_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'B']], ['R', 'W'], 2)

cma_2x2x2x2 = make_factor(sort_2x2x2x2_port, ['C', 'A'], 3)
cma_s_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'S']], ['C', 'A'], 3)
cma_b_2x2x2x2 = make_factor(sort_2x2x2x2_port[[x for x in sort_2x2x2x2_port.columns if x[0] == 'B']], ['C', 'A'], 3)


def get_port(fac1, fac2, n, monthly_rtn, market_cap, get_rtn_f, size=None):

    q_list = [str(x) for x in range(1, n+1)]
    result = pd.DataFrame(index=monthly_rtn.index, columns=[str(x)+str(y) for x in range(1, n+1) for y in range(1, n+1)])

    decile_list = pd.DataFrame(index=monthly_rtn.index, columns=[str(x)+str(y) for x in range(1, n+1) for y in range(1, n+1)]).astype('str')
    decile_list[:] = ''

    quint1 = fac1.apply(lambda x: pd.qcut(x.rank(method='first'), len(q_list), labels=q_list), axis=1).astype('str').replace('nan', np.nan)
    quint2 = fac2.apply(lambda x: pd.qcut(x.rank(method='first'), len(q_list), labels=q_list), axis=1).astype('str').replace('nan', np.nan)

    decile_list = quint1 + quint2

    if not isinstance(size, type(None)):
        decile_list = size + decile_list

    tmp_df = decile_list.apply(lambda x: get_rtn_f(x.dropna(), result, monthly_rtn, market_cap), axis=1)

    return result.dropna(axis=0)



def get_rtn_ratio(x, result, ratio, market_cap):
    start_d = str(x.name.year)+'-06'
    unique_v = x.unique()
    for g in unique_v:
        tic_list = x[x == g].index
        tmp_mv = market_cap.loc[x.name.strftime('%Y-%m'), tic_list].iloc[0]
        tmp_rtn = ratio.loc[start_d, tic_list]
        tmp_mv /= tmp_mv.sum()
        result.loc[start_d,g] = (tmp_rtn*tmp_mv).sum(skipna=True, axis=1)

    return 'Done!'


size_bm_5x5 = get_port(market_cap, bm, 5, monthly_rtn, market_cap, get_rtn)
size_op_5x5 = get_port(market_cap, op, 5, monthly_rtn, market_cap, get_rtn)
size_inv_5x5 = get_port(market_cap, inv, 5, monthly_rtn, market_cap, get_rtn)
#size_xrd_5x5 = get_port(market_cap, xrd_me, 5, monthly_rtn, market_cap, get_rtn)


result = pd.DataFrame(index=monthly_rtn.index, columns=[str(x) for x in range(1, 6)])
#quin_list = xrd_me.apply(lambda x: pd.qcut(x.rank(method='first'), 5, labels=['1','2','3','4','5']), axis=1).astype('str').replace('nan', np.nan)

aa = fac_port_rtn(quin_list , monthly_rtn, market_cap)
aa2 = aa['5'] - aa['1']

import statsmodels.api as sm
x = ff5_monthly.drop('Rf', axis=1)
x = sm.add_constant(x)

model = sm.OLS(aa2.astype('float').loc['2000':], x.astype('float').loc['2000':]).fit(cov_type='HAC', cov_kwds={'maxlags':12})
model.summary()



size_bm_5x5.loc['2000':].mean().sort_values(ascending=False)


y = ff5_monthly['HML']
x = ff5_monthly.drop(['HML','Rf'] ,axis=1)

size_bm_5x5_bm = get_port(market_cap, bm, 5, bm, market_cap, get_rtn_ratio)


size_bm_5x5.mean()

three_alpha = pd.DataFrame(index=['1','2','3','4','5'], columns = ['1','2','3','4','5','t1','t2','t3','t4','t5'])

five_alpha = pd.DataFrame(index=['1','2','3','4','5'], columns = ['1','2','3','4','5','t1','t2','t3','t4','t5'])
five_h = pd.DataFrame(index=['1','2','3','4','5'], columns = ['1','2','3','4','5','t1','t2','t3','t4','t5'])
five_r = pd.DataFrame(index=['1','2','3','4','5'], columns = ['1','2','3','4','5','t1','t2','t3','t4','t5'])
five_c = pd.DataFrame(index=['1','2','3','4','5'], columns = ['1','2','3','4','5','t1','t2','t3','t4','t5'])




#table 4

from scipy.stats import ttest_1samp


def get_stat(fac):
    mean = np.round(fac.mean(),4) * 100
    mean.name = 'MEAN'

    tstat = fac.apply(lambda x: np.round(ttest_1samp(x, 0)[0],2))
    tstat.name = 't-Stat'

    std = np.round(fac.std(),4) * 100
    std.name = 'Std'

    result = pd.concat([mean, std, tstat],axis=1).T

    return result

ff5_2x2 = pd.concat([smb_2x2, hml_2x2, rmw_2x2, cma_2x2],axis=1)
ff5_2x2.columns = ['SMB_2x2', 'HML_2x2', 'RMW_2x2', 'CMA_2x2']
ff5_2x2x2x2 = pd.concat([smb_2x2x2x2, hml_2x2x2x2, rmw_2x2x2x2, cma_2x2x2x2],axis=1)
ff5_2x2x2x2.columns = ['SMB_2x2x2x2', 'HML_2x2x2x2', 'RMW_2x2x2x2', 'CMA_2x2x2x2']

hml_2x3_sb = pd.concat([hml_s, hml_b, hml_s-hml_b],axis=1)
hml_2x3_sb.columns = ['HML_S', 'HML_B', 'HML_S-B']
hml_2x2_sb = pd.concat([hml_s_2x2, hml_b_2x2, hml_s_2x2-hml_b_2x2],axis=1)
hml_2x2_sb.columns = ['HML_S_2x2', 'HML_B_2x2', 'HML_S-B_2x2']
hml_2x2x2x2_sb = pd.concat([hml_s_2x2x2x2, hml_b_2x2x2x2, hml_s_2x2x2x2-hml_b_2x2x2x2],axis=1)
hml_2x2x2x2_sb.columns = ['HML_S_2x2x2x2', 'HML_B_2x2x2x2', 'HML_S-B_2x2x2x2']

rmw_2x3_sb = pd.concat([rmw_s, rmw_b, rmw_s-rmw_b],axis=1)
rmw_2x3_sb.columns = ['rmw_S', 'rmw_B', 'rmw_S-B']
rmw_2x2_sb = pd.concat([rmw_s_2x2, rmw_b_2x2, rmw_s_2x2-rmw_b_2x2],axis=1)
rmw_2x2_sb.columns = ['rmw_S_2x2', 'rmw_B_2x2', 'rmw_S-B_2x2']
rmw_2x2x2x2_sb = pd.concat([rmw_s_2x2x2x2, rmw_b_2x2x2x2, rmw_s_2x2x2x2-rmw_b_2x2x2x2],axis=1)
rmw_2x2x2x2_sb.columns = ['rmw_S_2x2x2x2', 'rmw_B_2x2x2x2', 'rmw_S-B_2x2x2x2']

cma_2x3_sb = pd.concat([cma_s, cma_b, cma_s-cma_b],axis=1)
cma_2x3_sb.columns = ['cma_S', 'cma_B', 'cma_S-B']
cma_2x2_sb = pd.concat([cma_s_2x2, cma_b_2x2, cma_s_2x2-cma_b_2x2],axis=1)
cma_2x2_sb.columns = ['cma_S_2x2', 'cma_B_2x2', 'cma_S-B_2x2']
cma_2x2x2x2_sb = pd.concat([cma_s_2x2x2x2, cma_b_2x2x2x2, cma_s_2x2x2x2-cma_b_2x2x2x2],axis=1)
cma_2x2x2x2_sb.columns = ['cma_S_2x2x2x2', 'cma_B_2x2x2x2', 'cma_S-B_2x2x2x2']


smb_tot = pd.concat([smb_ff5, smb_2x2, smb_2x2x2x2],axis=1)
smb_tot.columns = ['2x3','2x2','2x2x2x2']
hml_tot = pd.concat([hml, hml_2x2, hml_2x2x2x2],axis=1)
hml_tot.columns = ['2x3','2x2','2x2x2x2']
rmw_tot = pd.concat([rmw, rmw_2x2, rmw_2x2x2x2],axis=1)
rmw_tot.columns = ['2x3','2x2','2x2x2x2']
cma_tot = pd.concat([cma, cma_2x2, cma_2x2x2x2],axis=1)
cma_tot.columns = ['2x3','2x2','2x2x2x2']

((1+ff5_monthly.drop('Rf',axis=1).loc['2000':]).cumprod()-1).plot()


t4_stat1 = pd.concat([get_stat(ff5_monthly.drop('Rf',axis=1)), get_stat(ff5_2x2), get_stat(ff5_2x2x2x2)], axis=1)
t4_stat_hml = pd.concat([get_stat(hml_2x3_sb), get_stat(hml_2x2_sb), get_stat(hml_2x2x2x2_sb)], axis=1)
t4_stat_rmw = pd.concat([get_stat(rmw_2x3_sb), get_stat(rmw_2x2_sb), get_stat(rmw_2x2x2x2_sb)], axis=1)
t4_stat_cma = pd.concat([get_stat(cma_2x3_sb), get_stat(cma_2x2_sb), get_stat(cma_2x2x2x2_sb)], axis=1)

print(t4_stat_cma.to_string())
cma

smb_tot.astype('float').corr()
hml_tot.astype('float').corr()
rmw_tot.astype('float').corr()
cma_tot.astype('float').corr()

ff5_monthly.drop('Rf',axis=1).astype('float').corr()
pd.concat([mkt_fac,ff5_2x2],axis=1).astype('float').corr()
pd.concat([mkt_fac,ff5_2x2x2x2],axis=1).astype('float').corr()

t4_hml_2x3_stat = get_stat(hml_2x3_sb)
t4_hml_2x2_stat = get_stat(hml_2x2_sb)
t4_hml_2x3_stat

