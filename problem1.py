from pyspark import SparkConf, SparkContext;
import pyspark.sql as sql;
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
    sqlContext = sql.SQLContext( sc );

    crimes = sc.textFile( 'hdfs://wolf.iems.northwestern.edu/user/huser72/crimes.csv' );

    # Grab the header from the data
    crimesHeader = crimes.first( ).split( ',' );

    # Add index to each line of file and filter out the header, parse the CSV lines as separate fields,
    # and select the ID and date fields
    crimesData = crimes.zipWithIndex( ) \
                       .filter( lambda pair: pair[ 1 ] > 0 ) \
                       .map( csvParse ) \
                       .map( lambda row: ( int( row[ 0 ] ), row[ 2 ] ) ) \
                       .cache( );

    # Create a StructField with the right data type to for each column to be retained
    fields = [ sql.types.StructField( crimesHeader[ i ], sql.types.StringType( ), True ) for i in [ 0, 2 ] ];
    fields[ 0 ].dataType = sql.types.IntegerType( );

    # Create schema object, and convert crimes data into SQL schema using it
    schema = sql.types.StructType( fields );
    schemaCrime = sqlContext.applySchema( crimesData, schema );

    schemaCrime.registerTempTable( 'crimes' );

    # Query to fetch average crimes for each month across all years
    result = sqlContext.sql( 'select substr( Date, 1, 2 ) as month, count( ID ) / count( distinct substr( Date, 7, 4 ) ) as crime_count ' \
                             'from crimes ' \
                             'group by substr( Date, 1, 2 )' );
    counts = result.map( lambda row: ( row.month, row.crime_count ) );

    for c in counts.collect( ):
        print 'Month', c[ 0 ], '\tCount', c[ 1 ];

    sc.stop( );
