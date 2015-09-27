from pyspark import SparkConf, SparkContext;
import csv;
import datetime as dt;
from dateutil import relativedelta as rd;
import math;

def csvParse( t ):
    '''
    Takes a tuple and converts the first element from a CSV string to a list object containing
    each field from the CSV string as an element.

    @param  tup tuple whose first element is a comma-separated (CSV) string
    @return list of strings where each element is a field from the CSV string
    '''
    line = t[ 0 ];
    reader = csv.reader( [ line ] );
    return list( reader )[ 0 ];

if __name__ == '__main__':
    sc = SparkContext( );
    
    # Read the data, add index to each line and filter out the header, and convert each CSV line to different fields
    crimes = sc.textFile( 'hdfs://wolf.iems.northwestern.edu/user/huser72/crimes.csv' ) \
               .zipWithIndex( ) \
               .filter( lambda t: t[ 1 ] > 0 ) \
               .map( csvParse );

    # Grab the date and district, filter out blank districts and convert district number to integer
    filteredCrimes = crimes.map( lambda row: ( dt.datetime.strptime( row[ 2 ], '%m/%d/%Y %I:%M:%S %p' ), row[ 11 ] ) ) \
                           .filter( lambda row: row[ 1 ] != '' ) \
                           .map( lambda row: ( row[ 0 ], int( row[ 1 ] ) ) );

    # Split the datasets by mayors Daley and Rahm
    crimesDaley = filteredCrimes.filter( lambda row: row[ 0 ].year < 2011 or ( row[ 0 ].year == 2011 and row[ 0 ].month <= 5 ) ) \
                                .cache( );
    crimesRahm  = filteredCrimes.subtract( crimesDaley ) \
                                .cache( );

    # Find the start and end dates for Daley and Rahm in the dataset
    minDateDaley = crimesDaley.map( lambda row: row[ 0 ] ) \
                              .reduce( min );
    maxDateDaley = crimesDaley.map( lambda row: row[ 0 ] ) \
                              .reduce( max );
    minDateRahm  = crimesRahm.map(  lambda row: row[ 0 ] ) \
                             .reduce(  min );
    maxDateRahm  = crimesRahm.map(  lambda row: row[ 0 ] ) \
                             .reduce(  max );

    # Calculate the fractoinal years for Daley and Rahm
    yearsDaley = rd.relativedelta( maxDateDaley, minDateDaley ).years + float( rd.relativedelta( maxDateDaley, minDateDaley ).months ) / 12;
    yearsRahm  = rd.relativedelta( maxDateRahm,  minDateRahm  ).years + float( rd.relativedelta( maxDateRahm,  minDateRahm  ).months ) / 12;

    # For Daley and Rahm, count total crimes by district, filter out districts with very low counts and normalize counts
    # by dividing by the years each mayor
    districtCrimesDaley = crimesDaley.map( lambda row: ( row[ 1 ], 1 ) ) \
                                     .reduceByKey( lambda x, y: x + y ) \
                                     .filter( lambda row: row[ 1 ] > 100 ) \
                                     .map( lambda row: ( row[ 0 ], row[ 1 ] / yearsDaley ) );
    districtCrimesRahm  = crimesRahm.map(  lambda row: ( row[ 1 ], 1 ) ) \
                                    .reduceByKey(  lambda x, y: x + y ) \
                                    .filter( lambda row: row[ 1 ] > 100 ) \
                                    .map(  lambda row: ( row[ 0 ], row[ 1 ] / yearsRahm  ) );

    # Join the average crime per district for both mayors and compute the difference
    joinedDistrictCrimes = districtCrimesDaley.join( districtCrimesRahm );
    deltaDistrictCrimes = joinedDistrictCrimes.map( lambda row: row[ 1 ][ 0 ] - row[ 1 ][ 1 ] ) \
                                              .cache( );

    # Perform a paired t-test and compute the t-score
    meanDiff = deltaDistrictCrimes.mean( );
    sdDiff = deltaDistrictCrimes.sampleStdev( );
    n = deltaDistrictCrimes.count( );
    tstat = meanDiff / ( sdDiff / math.sqrt( n ) );
    
    sc.stop( );
    print 'Difference Mean:\t', meanDiff, '\nDifference Std Dev:\t', sdDiff, '\nN:\t\t\t', n, '\nT-statistic:\t\t', tstat;

