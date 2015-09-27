from pyspark import SparkConf, SparkContext;
from pyspark.mllib.linalg import Vectors;
from pyspark.mllib.stat import Statistics;
import csv;
import numpy as np;


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
               .map( csvParse );

    # Grab the Beat and Year information for each crime record, count the number of crimes for each
    # beat-year combination, and cache the dataset
    crimeCounts = crimes.map( lambda row: ( ( row[ 10 ], int( row[ 17 ] ) ), 1 ) ) \
                        .reduceByKey( lambda x, y: x + y ) \
                        .cache( );

    # Compute all possible beat-year combinations
    allBeatYears = crimeCounts.keys( ) \
                              .flatMap( lambda key: [ ( key[ 0 ], y ) for y in range( 2001, 2016 ) ] );

    # Compute the missing beat-year combinations in crimeCounts (beat-year combinations with 0 crime)
    missingBeatYears = allBeatYears.subtract( crimeCounts.keys( ) ).distinct( );

    # Add missing beat-year combinations to crimeCounts with 0 count
    fullCrimeCounts = crimeCounts.union( missingBeatYears.map( lambda row: ( row, 0 ) ) );

    # Map each year to all beats and their corresponding crime counts for that year, and sort the counts 
    # by beat
    groupedCounts = fullCrimeCounts.map( lambda row: ( row[ 0 ][ 1 ], ( row[ 0 ][ 0 ], row[ 1 ] ) ) ) \
                                   .groupByKey( ) \
                                   .mapValues( lambda val: sorted( list( val ), key = lambda t: t[ 0 ] ) );

    # Create a list of all beats
    beats = [ elem[ 0 ] for elem in groupedCounts.values( ).first( ) ];

    # Create Vectors object for beat-wise crime counts for each year. The RDD has one Vectors object per
    # year and each Vectors object has that year's crime counts for each beat, sorted by beats
    vectorCounts = groupedCounts.values( ) \
                                .map( lambda row: Vectors.dense( [ elem[ 1 ] for elem in row ] ) );

    # Compute correlation between all beats for yearly crime counts
    corr = Statistics.corr( vectorCounts, method = 'pearson' );
    sc.stop( );
    
    # Fill the diagonal of correlation matrix with 0's
    corr.flags[ 'WRITEABLE' ] = True;
    np.fill_diagonal( corr, 0.0 );

    # Get the 10 largest correlation values from the matrix. The correlation matrix is symmetric so 
    # we take the largest 20 and step by 2. Finally, the index of the corresponding beat pairs for
    # top 10 correlation values is obtained.
    sortOrder = corr.argsort( axis = None );
    indices = np.unravel_index( sortOrder[ -20::2 ], corr.shape  );

    # The corresponding beats names are obtained for the top 10 correlated beat pairs
    topBeatPairs = [ ( beats[ i ], beats[ j ] ) for i, j in zip( indices[ 0 ], indices[ 1 ] ) ];

    for i, j in topBeatPairs:
        print i, j;

