from pyspark import SparkContext;
import csv;

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

    # Grab the block and year for each row and retain only those crimes that occurred after 2011
    crimesSubset = crimes.map( lambda row: ( row[ 3 ], int( row[ 17 ] ) ) ) \
                         .filter( lambda row: row[ 1 ] > 2011 );

    # Count all crimes per block in the subset
    blockCrimes = crimesSubset.map( lambda row: ( row[ 0 ], 1 ) ) \
                              .reduceByKey( lambda x, y: x + y );

    # Flip the KV pair (block, count) to (count, block), perform descending sort by counts, and flip back
    sortedCrimes = blockCrimes.map( lambda row: ( row[ 1 ], row[ 0 ] ) ) \
                              .sortByKey( False ) \
                              .map( lambda row: ( row[ 1 ], row[ 0 ] ) );
    # Get the top 10 blocks
    topBlocks = sortedCrimes.take( 10 );

    sc.stop( );

    for pair in topBlocks:
        print "Block:", pair[ 0 ], "\tCrime Count:", pair[ 1 ];

