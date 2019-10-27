import pandas as pd

#These are features that have less than 50% missing data and
#are not corresponding to limit or error of a measurement and
#are not present in the unimportantFeatures.txt file

selectedFeats=['pl_name', 'pl_controvflag', 'pl_pnum', 'pl_orbper', 'pl_orbsmax', 'pl_radj', 'pl_ttvflag', 'pl_kepflag', 'pl_k2flag', 'ra', 'dec', 'st_dist', 'st_optmag', 'st_optband', 'gaia_gmag', 'st_teff', 'st_mass', 'st_rad', 'pl_tranflag', 'pl_rvflag', 'pl_imgflag', 'pl_astflag', 'pl_omflag', 'pl_cbflag', 'pl_angsep', 'pl_rade', 'pl_rads', 'pl_trandur', 'pl_tranmid', 'pl_ratror', 'pl_mnum', 'pl_st_npar', 'pl_st_nref', 'st_rah', 'st_glon', 'st_glat', 'st_elon', 'st_elat', 'gaia_plx', 'gaia_dist', 'st_pmra', 'st_pmdec', 'st_pm', 'gaia_pmra', 'gaia_pmdec', 'gaia_pm', 'st_logg', 'st_metfe', 'st_j', 'st_h', 'st_k', 'st_wise1', 'st_wise2', 'st_wise3', 'st_wise4', 'st_jmh2', 'st_hmk2', 'st_jmk2']

exos=pd.read_csv('exoplanets.csv')

#These are the exoplanets(ie rows) that have more than 25% missing data.
Morethan25NA=['CFBDSIR J145829+101343 b', 'DP Leo b', 'HD 41004 B b', 'IC 4651 9122 b', 'KMT-2016-BLG-1107L b', 'KMT-2016-BLG-1397L b', 'KMT-2016-BLG-1820L b', 'KMT-2016-BLG-2142L b', 'KMT-2017-BLG-1038L b', 'KMT-2017-BLG-1146L b', 'MOA-2007-BLG-192L b', 'MOA-2007-BLG-400L b', 'MOA-2008-BLG-310L b', 'MOA-2008-BLG-379L b', 'MOA-2009-BLG-266L b', 'MOA-2009-BLG-319L b', 'MOA-2009-BLG-387L b', 'MOA-2010-BLG-073L b', 'MOA-2010-BLG-117L b', 'MOA-2010-BLG-328L b', 'MOA-2010-BLG-353L b', 'MOA-2010-BLG-477L b', 'MOA-2011-BLG-028L b', 'MOA-2011-BLG-262L b', 'MOA-2011-BLG-291L b', 'MOA-2011-BLG-293L b', 'MOA-2011-BLG-322L b', 'MOA-2012-BLG-006L b', 'MOA-2012-BLG-505L b', 'MOA-2013-BLG-605L b', 'MOA-2015-BLG-337L b', 'MOA-2016-BLG-227L b', 'MOA-2016-BLG-319L b', 'MOA-bin-1L b', 'MXB 1658-298 b', 'OGLE-2003-BLG-235L b', 'OGLE-2005-BLG-071L b', 'OGLE-2005-BLG-169L b', 'OGLE-2005-BLG-390L b', 'OGLE-2006-BLG-109L b', 'OGLE-2006-BLG-109L c', 'OGLE-2007-BLG-349L AB c', 'OGLE-2007-BLG-368L b', 'OGLE-2008-BLG-092L b', 'OGLE-2008-BLG-355L b', 'OGLE-2011-BLG-0173L b', 'OGLE-2011-BLG-0251L b', 'OGLE-2011-BLG-0265L b', 'OGLE-2012-BLG-0026L b', 'OGLE-2012-BLG-0026L c', 'OGLE-2012-BLG-0358L b', 'OGLE-2012-BLG-0406L b', 'OGLE-2012-BLG-0563L b', 'OGLE-2012-BLG-0724L b', 'OGLE-2012-BLG-0950L b', 'OGLE-2013-BLG-0102L b', 'OGLE-2013-BLG-0132L b', 'OGLE-2013-BLG-0341L B b', 'OGLE-2013-BLG-1721L b', 'OGLE-2013-BLG-1761L b', 'OGLE-2014-BLG-0124L b', 'OGLE-2014-BLG-0676L b', 'OGLE-2014-BLG-1722L b', 'OGLE-2014-BLG-1722L c', 'OGLE-2014-BLG-1760L b', 'OGLE-2015-BLG-0051L b', 'OGLE-2015-BLG-0954L b', 'OGLE-2015-BLG-0966L b', 'OGLE-2016-BLG-0263L b', 'OGLE-2016-BLG-0613L AB b', 'OGLE-2016-BLG-1190L b', 'OGLE-2016-BLG-1195L b', 'OGLE-2017-BLG-0173L b', 'OGLE-2017-BLG-0373L b', 'OGLE-2017-BLG-0482L b', 'OGLE-2017-BLG-1140L b', 'OGLE-2017-BLG-1434L b', 'OGLE-2017-BLG-1522L b', 'OGLE-TR-113 b', 'OGLE-TR-132 b', 'PSR B1257+12 b', 'PSR B1257+12 c', 'PSR B1257+12 d', 'PSR B1620-26 b', 'PSR J1719-1438 b', 'PSR J2322-2650 b', 'SWEEPS-11 b', 'SWEEPS-4 b', 'TCP J05074264+2447555 b', 'UKIRT-2017-BLG-001L b', 'VHS J125601.92-125723.9 b', 'WISEP J121756.91+162640.2 A b']

#removing planets with more than 25% missing data
exos=exos[~exos.pl_name.isin(Morethan25NA)]

#only including those features that are in selectedFeats variable
exos=exos[selectedFeats]

def findHabitable(exos):
""" Print how many planets from the habitable planets catalogue consevative estimate are present in the exoplanet archive data"""
    habitable=['GJ 667 C c', 'GJ 667 C e', 'GJ 667 C f', 'Kepler-1229 b', 'Kepler-1652 b', 'Kepler-186 f', 'Kepler-442 b', 'Kepler-62 f', 'LHS 1140 b', 'Proxima Cen b', 'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g']
    print('num discovered habitable: ',len(habitable))
    missingHabitable=[]
    allPlanets=exos.pl_name.values
    
    for planet in habitable:
        if planet not in allPlanets:
            missingHabitable.append(n)
    if len(missingHabitable)>0:
        print('num habitbale in exos: ',str(len(habitable)-len(missingHabitable)))
        print('missing are:')
        print(missingHabitable)

    print('num habitbale in exos: ',str(len(habitable)-len(missingHabitable)))

#Removing all records that have mssing values in the st_optband column
exos=exos[~exos.st_optband.isnull()]
findHabitable(exos)

#Removing pl_name column and st_optband
nonCategorical=[x for x in exos.columns if x!="pl_name" and x!="st_optband"]
print("Num Features remaining ",len(nonCategorical))

exos=exos[nonCategorical]
print("Num Records remaining: ",len(exos))

exos.to_csv('selectedFeatures.csv')
