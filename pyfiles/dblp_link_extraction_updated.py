#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import requests
import scrapy
from scrapy.selector import Selector
from w3lib.html import remove_tags
from tqdm import tqdm
from difflib import get_close_matches, SequenceMatcher
import time 
import json
import argparse
#url='http://dblp.uni-trier.de/search/author/?q=""'
DBLP_BASE_URL = 'http://dblp.uni-trier.de'
AUTHOR_SEARCH_URL = DBLP_BASE_URL+"/search/author/"


def check_matches(response, name, aff):
    url=None
    if response:
        likely_match= similar_authors(response,'Likely')
        exact_match = similar_authors(response,'Exact')
        all_match   = similar_authors(response,'All')
        first_1000  = similar_authors(response,'First')
        if not (exact_match['names'] or likely_match['names'] or all_match['names'] or first_1000['names']):
            page_response     =   requests.get(response.url)
            page_response_sel =   Selector(page_response)
            if not page_response_sel.xpath("//div[@id='completesearch-authors']//p[contains(.,'no matches')]"):
                    url = response.url
        elif len(exact_match['names']) > 0:
            url = get_likely_match(name, exact_match, aff)
        elif len(likely_match['names']) > 0:
            url = get_likely_match(name, likely_match, aff)
        elif len(all_match['names']) < 10:
            url = get_likely_match(name, all_match, aff)
        else:
            url = None
    return url



def get_likely_match(name, info_dict, aff):
    #print(aff)
    #print(name)
    close_match = get_close_matches(name, info_dict['names'], 50)
    #print(close_match)
    #print(info_dict)
    if len(close_match)==1:
        close_name = close_match[0]
        #print(close_name)
        close_name_index = info_dict['names'].index(close_name)
        #print(close_name_index)
        url = info_dict['urls'][close_name_index]
    elif len(close_match) > 1 and len(info_dict['other_info']) > 0:
        aff_close_match = get_close_matches(aff, info_dict['other_info'],15)
        if len(aff_close_match):
            aff_close_match = aff_close_match[0]
            close_aff_index = info_dict['other_info'].index(aff_close_match)
            url = info_dict['urls'][close_aff_index]
        else:
            list_values = [(url,SequenceMatcher(None, aff, aff1).ratio()) for aff1, url in zip(info_dict['other_info'], info_dict['urls'])]
            sorted_values = sorted(list_values, key = lambda x: x[1])
            url = sorted_values[-1][0]
    elif len(close_match) > 1 and len(info_dict['other_info'])== 0:
        close_name1 = close_match[0]
        #print(close_name)
        close_name_index1 = info_dict['names'].index(close_name1)
        #print(close_name_index)
        url = info_dict['urls'][close_name_index1]
    else:
        list_values = [(url,SequenceMatcher(None, name, name1).ratio()) for name1, url in zip(info_dict['names'], info_dict['urls'])]
        sorted_values = sorted(list_values, key = lambda x: x[1])
        url = sorted_values[-1][0]
    #print(url)
    return url


def search_author(name, idd, aff):
    url=None
    try: 
        response = requests.get(AUTHOR_SEARCH_URL, params={'q':name})
        filename =f"../data/dblp_data/candidate_link_pages/{idd}.html"
        with open(filename, 'wb') as f:
            f.write(response.content)
        url=check_matches(response, name, aff)                                                                      
    except Exception as e:
        with open("../data/dblp_data/mgp_dblp_extract_error_id.txt", "a") as f1:
            f1.write(str(idd))
            f1.write("\n")
        response=None
        print(e)
        print(name)
    return url 


def similar_authors(response, string='All'):
    resp_selector=Selector(response)
    names_sel=resp_selector.xpath("//p[starts-with(text(),'{}')]/following-sibling::ul[1]/li//span[@itemprop='name']".format(string))
    names=[remove_tags(name) for name in names_sel.getall()]
    other_sel=resp_selector.xpath("//p[starts-with(text(),'{}')]/following-sibling::ul[1]/li/small[1]".format(string))
    other_info=[remove_tags(aff) for aff in other_sel.getall()]
    urls=resp_selector.xpath("//p[starts-with(text(),'{}')]/following-sibling::ul[1]/li//a/@href".format(string)).getall()
    return {'names':names, 'other_info': other_info, 'urls':urls}



def add_dblp_link(file, start_index, last_index):
    temp_dict={}
    temp=[]
    people=pd.read_csv(file, sep=',', lineterminator='\n', low_memory=False)
    people = people[['Id', 'Name', 'Year', 'University', 'Country', 'Title', 'MSC', 'BIO','MSN']]
    people.fillna('', inplace=True)
    
    #open changed1
    people['mod_Country']    = people['Country'].replace("UnitedStates","USA")#changed1 date - 26-05-2022
    people['mod_Country']    = people['mod_Country'].replace("UnitedKingdom","UK")
    people['mod_University'] = people['University'].str.replace("University|\t"," ")#changed1
    people['mod_University'] = people['mod_University'].str.replace("  "," ")#changed1
    #closed changed1

    people['University_country'] = people['mod_University']+","+ people['mod_Country']
    #people = people[people['Id'].isin(idd)]
    people['Name']=people['Name'].replace(to_replace = ' +', value = ' ', regex = True)
    people = people.iloc[start_index:last_index]
    #people = people.sample(10)
    file_count = 2
    count=0
    try:
        for name, idd, aff in tqdm(zip(people['Name'].values, people['Id'].values, people['University_country'].values), total = (last_index-start_index)):
            url = search_author(name, idd, aff)
            temp.append(url)
            temp_dict[int(idd)] = url
            count+=1
            if count % 1000 == 0:
                time.sleep(5)
                print("processed rows {}".format(count))
                temp_dict['last_index'] = start_index+count
                with open(f'../data/dblp_data/mgp_dblp_results_{file_count}.json', 'w') as fp:
                    json.dump(temp_dict, fp)
    except Exception as e1:
        print(f"error id : {idd}, name: {name}")
        print("Total Count:{}".format(len(temp)))
        print(e1)
        with open('../data/dblp_data/temp_dblp_results.json', 'w') as fp:
            json.dump(temp_dict, fp)
        exit()
    people['dblp_link_new'] = temp
    filename = f"../data/dblp_data/mgpnodeList_with_year_completion_updated_dblpLink_{start_index}_{last_index}.csv"
    people.to_csv(filename, sep=',')
    print("finished")
    return



if __name__=="__main__":
    start = time.time()
    print("Extraction started...")
    parser = argparse.ArgumentParser(description='DBLP Link Extraction')
    parser.add_argument('-si','--start_index', help='mgp start index', required=True)
    parser.add_argument('-li','--last_index', help='mgp last index', required=True)
    args = vars(parser.parse_args())

    start_index = int(args['start_index'])
    last_index =  int(args['last_index'])
    print(f"started from {start_index} to {last_index}")
    add_dblp_link("../data/mgp_data/mgpnodeList_with_year_completion_updated_with_complete_dblp_link.csv", start_index, last_index)
    print("Process finished")
    stop = time.time()
    print('Time (in hr): ', (stop - start)/3600)
