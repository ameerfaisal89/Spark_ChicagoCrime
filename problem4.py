from pyspark import SparkConf, SparkContext;
from pyspark.mllib.linalg import Vectors;
from pyspark.mllib.stat import Statistics;
import csv;
import datetime as dt;

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

    # Read the data, add index to each line, filter out the header and parse each CSV line as a separate field
    crimes = sc.textFile( 'hdfs://wolf.iems.northwestern.edu/user/huser72/crimes.csv' ) \
               .zipWithIndex( ) \
               .filter( lambda row: row[ 1 ] > 0 ) \
               .map( csvParse );

    arrests = crimes.filter( lambda row: row[ 8 ] == 'true' ) \
                    .map( lambda row: ( dt.datetime.strptime( row[ 2 ], '%m/%d/%Y %I:%M:%S %p' ), row[ 5 ] ) ) \
                    .cache( );

    totalArrests = arrests.count( );

    arrestsHour = arrests.map( lambda row: row[ 0 ].hour ) \
                         .histogram( list( range( 25 ) ) );

    arrestsWeek = arrests.map( lambda row: row[ 0 ].weekday( ) ) \
                         .histogram( list( range( 8 ) ) );

    arrestsMonth = arrests.map( lambda row: row[ 0 ].month ) \
                          .histogram( list( range( 1, 14 ) ) );

    homicide = arrests.filter( lambda row: row[ 1 ] == 'HOMICIDE' ) \
                       .cache( );

    totalHomicide = homicide.count( );

    homicideHour = homicide.map( lambda row: row[ 0 ].hour ) \
                           .histogram( list( range( 25 ) ) );

    homicideWeek = homicide.map( lambda row: row[ 0 ].weekday( ) ) \
                           .histogram( list( range( 8 ) ) );

    homicideMonth = homicide.map( lambda row: row[ 0 ].month ) \
                            .histogram( list( range( 1, 14 ) ) );

    robbery = arrests.filter( lambda row: row[ 1 ] in [ 'ROBBERY', 'BURGLARY', 'THEFT' ] ) \
                       .cache( );

    totalRobbery = robbery.count( );

    robberyHour = robbery.map( lambda row: row[ 0 ].hour ) \
                             .histogram( list( range( 25 ) ) );

    robberyWeek = robbery.map( lambda row: row[ 0 ].weekday( ) ) \
                             .histogram( list( range( 8 ) ) );

    robberyMonth = robbery.map( lambda row: row[ 0 ].month ) \
                              .histogram( list( range( 1, 14 ) ) );

    sc.stop( );
    
for x in [ arrestsHour, arrestsWeek, arrestsMonth, homicideHour, homicideWeek, homicideMonth, robberyHour, robberyWeek, robberyMonth ]:
    print '\n';
    for i, val in enumerate( x[ 1 ] ):
        print i, '\t', val;
