from pyspark import SparkConf, SparkContext;
from pyspark.mllib.linalg import Vectors;
from pyspark.mllib.stat import Statistics;
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD;
from pyspark.mllib.tree import RandomForest;
from pyspark.mllib.util import MLUtils;

import csv;
import numpy as np;
import datetime as dt;
from dateutil import relativedelta as rd;
import math;

def csvParse( tup ):
    '''
    Takes a tuple and converts the first element from a CSV string to a list object containing
    each field from the CSV string as an element.

    @param  tup tuple whose first element is a comma-separated (CSV) string
    @return list of strings where each element is a field from the CSV string
    '''
    line = tup[ 0 ];
    reader = csv.reader( [ line ] );
    return list( reader )[ 0 ];

if __name__ == '__main__':
    sc = SparkContext( );
    
    # Load CSV file from HDFS, append index to each line of file and filter out the header (index 0),
    # and split each field of the file into a separate element.
    crimes = sc.textFile( 'hdfs://wolf.iems.northwestern.edu/user/huser72/crimes.csv' ) \
               .zipWithIndex( ) \
               .filter( lambda tup: tup[ 1 ] > 0 ) \
               .map( csvParse ) \
               .map( lambda row: ( row[ 10 ], dt.datetime.strptime( row[ 2 ], '%m/%d/%Y %I:%M:%S %p' ) ) ) \
               .cache( );
    
    #minYear = crimes.map( lambda row: row[ 1 ].year ) \
    #                .min( );
    
    # Obtain the current year
    maxYear = crimes.map( lambda row: row[ 1 ].year ) \
                    .max( );
    #yearWeeks = sc.parallelize( [ ( y, w ) for y in range( minYear, maxYear ) for w in range( 0, 53 ) ] );
    
    beats = crimes.map( lambda row: row[ 0 ] ) \
                  .distinct( ) \
                  .zipWithIndex( ) \
                  .cache( );
    
    # Mapping each beat to its index
    beatsDict = dict( beats.collect( ) );
    
    # Compute the number of weekly crime events for each beat between 2001 and 2014
    crimeCounts = crimes.filter( lambda row: row[ 1 ].year < maxYear ) \
                        .map( lambda row: ( ( beatsDict[ row[ 0 ] ], row[ 1 ].year, row[ 1 ].isocalendar( )[ 1 ] - 1 ), 1 ) ) \
                        .reduceByKey( lambda x, y: x + y ) \
                        .cache( );
    
    crimes.unpersist( );
    
    # Obtain all year-week combinations in the dataset
    yearWeeks = crimeCounts.map( lambda row: ( row[ 0 ][ 1 ], row[ 0 ][ 2 ] ) ) \
                           .distinct( );
    # Generate all possible beat-year-week combinations from 2001 to 2014
    allBeatYearWeeks = beats.values( ) \
                            .cartesian( yearWeeks ) \
                            .map( lambda row: ( row[ 0 ], row[ 1 ][ 0 ], row[ 1 ][ 1 ] ) );
    
    # Determine missing beat-year-week combinations in the dataset and insert them with
    # count 0 to the crime counts
    missingBeatYearWeeks = allBeatYearWeeks.subtract( crimeCounts.keys( ) );
    allCrimeCounts = crimeCounts.union( missingBeatYearWeeks.map( lambda row: ( row, 0 ) ) );
    
    # Load the historical temperature for the city and filter it for the years 2001 to 2014
    temperature = sc.textFile( 'hdfs://wolf.iems.northwestern.edu/user/huser72/chicago_temperature.txt' ) \
                    .map( lambda line: [ float( i ) for i in line.split( ) ] ) \
                    .filter( lambda row: row[ 2 ] > 2000 and row[ 2 ] < 2015 ) \
                    .map( lambda row: ( dt.date( row[ 2 ], row[ 0 ], row[ 1 ] ), row[ 3 ] ) );
    
    # Compute the average weekly temperature for each year
    avgTemperature = temperature.map( lambda row: ( ( row[ 0 ].year, row[ 0 ].isocalendar( )[ 1 ] ), ( row[ 1 ], 1 ) ) ) \
                                .reduceByKey( lambda x, y: ( x[ 0 ] + y[ 0 ], x[ 1 ] + y[ 1 ] ) ) \
                                .mapValues( lambda val: val[ 0 ] / val[ 1 ] );
    
    # Join the crime counts and average weekly temperature datasets, using year-week as key,
    # unnest each row to a flat list, drop the year variable and convert to a LabeledPoint object
    joinedData = allCrimeCounts.map( lambda row: ( ( row[ 0 ][ 1 ], row[ 0 ][ 2 ] ), ( row[ 0 ][ 0 ], row[ 1 ] ) ) ) \
                               .join( avgTemperature ) \
                               .map( lambda row: [ item for sublist in row for item in sublist ] ) \
                               .map( lambda row: LabeledPoint( row[ 2 ][ 1 ], [ row[ 2 ][ 0 ], row[ 1 ], row[ 3 ] ] ) ) \
                               .cache( );
    
    crimeCounts.unpersist( );
    
    # Split the crime counts into training and test datasets
    ( training, test ) = joinedData.randomSplit( ( 0.7, 0.3 ) );
    
    # Categorical features dictionary
    featuresInfo = { 0: len( beatsDict ), 1: 53 };
    
    # Train a Random Forest model to predict crimes
    model = RandomForest.trainRegressor( training, categoricalFeaturesInfo = featuresInfo,
                                         numTrees = 5, featureSubsetStrategy = "auto",
                                         impurity = 'variance', maxDepth = 10, maxBins = len( beatsDict ) );
    
    # Measure the model performance on test dataset
    predictions = model.predict( test.map( lambda x: x.features ) ) \
                       .cache( );
    
    meanCrimes = test.map( lambda x: x.label ).mean( );
    labelsAndPredictions = test.map( lambda x:  x.label ).zip( predictions );
    testMSE = labelsAndPredictions.map( lambda ( v, p ): ( v - p ) * ( v - p ) ).sum( ) / float( test.count( ) );
    testSSE = labelsAndPredictions.map( lambda ( v, p ): ( v - p ) * ( v - p ) ).sum( );
    testSST = labelsAndPredictions.map( lambda ( v, p ): ( v - meanCrimes ) * ( v - meanCrimes ) ).sum( );
    
    Rsq = 1 - testSSE / testSST;
    
    #### Predicting crimes for next week ####
    weekNum = 26;
    tempForecast = [ 67.0, 70.5, 68.0, 70.0, 70.0, 71.0, 76.5 ];
    
    # Average temperature for next week
    tempNextWeek = sum( tempForecast ) / len( tempForecast );
    
    # Inverse beats dictionary to map index to beat number
    beatsDictInverse = dict( ( v, k ) for k, v in beatsDict.items( ) );
    
    # Test dataset for each beat with next week's info
    nextWeek = sc.parallelize( tuple( [ ( beat, weekNum, tempNextWeek ) for beat in range( len( beatsDict ) ) ] ) );
    predictionsNextWeek = model.predict( nextWeek ) \
                               .zip( nextWeek.map( lambda row: beatsDictInverse[ row[ 0 ] ] ) ) \
                               .sortByKey( False );
    
    # Obtain the top 10 beats with highest likelihood of crime
    topCrimeBeats = predictionsNextWeek.take( 10 );
    
    sc.stop( );
    
    print( 'Test Mean Squared Error = ' + str( testMSE ) );
    print( 'Test R-squared = ' + str( Rsq ) );
    print( topCrimeBeats );
