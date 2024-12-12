# Use the Node.js 20 image as the base image
FROM #node:20

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy
COPY 

# Install dependencies
RUN 

# Copy the rest of the application code
COPY . .

# Build the TypeScript code
RUN 

# Expose the port the app runs on
EXPOSE 3000

# Command to run the app
CMD ["npm", "start"]