$(document).ready(function(){
	myGraph = new Graph({
              canvasId: 'QuadraticCanvas',
              minX: -10,
              minY: -10,
              maxX: 10,
              maxY: 10,
              unitsPerTick: 1
            });
	$('#TrainQuadratic').click(init);
});
const trueCoefficients = {a: -0.2, b: 0.9, c: 0.4};
	const trainingData = generateData(1000, trueCoefficients);
function generateData(numPoints, coeff, sigma = 0.3) {
  return tf.tidy(() => {
    const [a, b, c] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c)
    ];

    const xs = tf.randomUniform([numPoints], -4, 8);

    // Generate polynomial data
    const three = tf.scalar(3, 'int32');
    const ys = a.mul(xs.square())
      .add(b.mul(xs))
      .add(c)
	  .add(tf.randomNormal([numPoints], -1*sigma, sigma));;

    return {
      xs, 
      ys
    };
  })
}

	var myGraph;
   var randa = Math.random() * (1 - (-1) + 1) + (-1);
   var randb = Math.random() * (1 - (-1) + 1) + (-1);
   var randc = Math.random() * (1 - (-1) + 1) + (-1);
  
   const a = tf.variable(tf.scalar(randa));
   const b = tf.variable(tf.scalar(randb));       
   const c = tf.variable(tf.scalar(randc));
   a.print();
   b.print();
   c.print();
   
    
   function predict(x){
      return tf.tidy(() =>{
		redraw();
        return a.mul(x.square()) // + b * x ^ 2
      .add(b.mul(x)) // + c * x
      .add(c);
      });
   }

   function loss(predictions, labels){
      const meanSquareError = predictions.sub(labels).square().mean();
      return meanSquareError;
   }
      
   async function train(xs,ys,numIterations=10000000){
      const learningRate = $('#QuadraticLearningRate').val();
      const optimizer = tf.train.sgd(learningRate);
      
      for(let i=0; i < numIterations; i++){
          await sleep(10);
				 optimizer.minimize(() => {
				const predsYs = predict(xs);
				
				return loss(predsYs, ys);
			 });
		 
      }
   }
   
function init(){
	
	
	
    train(trainingData.xs,trainingData.ys,numIterations=10000000);

  
}
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function redraw(){
	myGraph.clear();
	myGraph = new Graph({
              canvasId: 'QuadraticCanvas',
              minX: -10,
              minY: -10,
              maxX: 10,
              maxY: 10,
              unitsPerTick: 1
            });
			
	myGraph.drawEquation(function(x) {
		return -0.2*x*x +0.9*x + 0.4 ;
	}, 'blue', 4);
   
    var coordinatesx = trainingData.xs.dataSync(); 
	
    var coordinatesy = trainingData.ys.dataSync(); 
	
    for(var i=0; i < coordinatesx.length; i++){
       myGraph.drawCoordinates(coordinatesx[i],coordinatesy[i], 'red', 1);
    }
	myGraph.drawEquation(function(x) {
          return a.dataSync()[0]*x*x + b.dataSync()[0]*x + c.dataSync()[0] ;
       }, 'green', 2);
  
}

